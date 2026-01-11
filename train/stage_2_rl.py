import torch
import transformers
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from utils.prompting import *
from utils.utils import write_json, append_jsonl, set_seed
import pandas as pd
from peft import (
    LoraConfig,
    get_peft_model,
)
import gc
import torch
import re
import statistics
from auto_gptq import exllama_set_max_input_length
from openai import OpenAI
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import Dataset
import ast
from rouge_score import rouge_scorer
from datasets.utils.logging import disable_progress_bar

wandb.init(mode="disabled")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# training hyperparams
batch_size = 1
num_epochs = 1
# learning_rate = 5e-6
learning_rate = 2e-7
# lora hyperparams
# lora_r = 8
# lora_alpha = 16
# lora_r = 16
# lora_alpha = 32
lora_r = 64
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
train_on_inputs = False  # if False, masks out inputs in loss
add_eos_token = False
eval_steps = 200
save_steps = 200
save_total_limit = 10
seed = 201
# debug_mode = False
debug_mode = True


def tokenize(tokenizer, prompt):
    result = tokenizer(
        prompt
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(TEMPLATE, tokenizer, data_point, response="", gen=False, eval=False):
    product_name = data_point['product_name']
    product_reviews = data_point['product_reviews']
    hist_vote_written = data_point['filtered_hist_vote_written']

    query = TEMPLATE.format(product_name=product_name, product_reviews=product_reviews,
                            hist_vote_written=hist_vote_written)
    if eval:
        msg = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(formatted_prompt)
    elif gen:
        msg = [{"role": "user", "content": query}]
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        tokenized_full_prompt = tokenize(formatted_prompt)
    else:
        msg = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(formatted_prompt)

    if not train_on_inputs:
        user_msg = [{"role": "user", "content": query}]
        user_prompt = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=False)
        tokenized_user_prompt = tokenize(user_prompt)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    tokenized_full_prompt['id'] = int(data_point['id'])
    return tokenized_full_prompt


# def make_generate_and_tokenize_prompt_gen(TEMPLATE, tokenizer):
#     def generate_and_tokenize_prompt_gen(data_point):
#         return generate_and_tokenize_prompt(TEMPLATE, tokenizer, data_point, gen=True)
#
#     return generate_and_tokenize_prompt_gen


def process_query_tensor(qt):
    i = qt.tolist().index(128000)
    return qt[i:]


def save_check_point():
    pass


def get_step_logs(stats, batch, rewards, step_i):
    logs = {}
    logs['step'] = step_i
    rewards = torch.stack(rewards)
    for k, v in stats.items():
        if not isinstance(v, np.ndarray):
            logs[k] = v
    logs["env/reward_mean"] = torch.mean(rewards).item()  # torch.mean(rewards).cpu().numpy().item()
    logs["env/reward_std"] = torch.std(rewards).item()  # torch.std(rewards).cpu().numpy().item()
    logs["env/reward_dist"] = rewards.tolist()  # rewards.cpu().numpy()
    return logs


def get_text_logs(pred, ans, epoch, step_i, idx_b):
    logs = []
    for p, a, idx in zip(pred, ans, idx_b):
        d = {'epoch': epoch, 'step': step_i, 'id': idx, 'output': p}
        logs.append(d)
        idx += 1
    return logs


def claim_extraction_from_summary(base_prompt, model, text):
    claim_split_prompt = base_prompt % (text)
    attempt = 0
    claim_split_response = None
    while attempt < 3 and claim_split_response == None:
        claim_split_response = get_completion(claim_split_prompt, model)
        attempt += 1

    if claim_split_response != None:
        claim_split_response = claim_split_response.replace("`", "").replace("json", "")
        generated_summary_kps = ast.literal_eval(claim_split_response)
    else:
        generated_summary_kps = []
    return generated_summary_kps


class CrossEncoderRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.device = torch.device("cpu")  # <<< force CPU

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, low_cpu_mem_usage=True)
        self.dropout = nn.Dropout(
            self.config.hidden_dropout_prob if hasattr(self.config, "hidden_dropout_prob") else 0.1)
        self.head = nn.Linear(self.config.hidden_size, 1)  # scalar

        # Ensure whole module is on CPU (paranoia; everything should already be)
        self.to(self.device)
        self.eval()  # inference by default for a reward model

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if input_ids.device.type != "cpu":
            input_ids = input_ids.to("cpu")
        if attention_mask is not None and attention_mask.device.type != "cpu":
            attention_mask = attention_mask.to("cpu")
        if token_type_ids is not None and token_type_ids.device.type != "cpu":
            token_type_ids = token_type_ids.to("cpu")

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # pool: prefer CLS token; if pooler exists (BERT), you can use pooler_output
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]  # CLS
        x = self.dropout(pooled)
        raw = self.head(x).squeeze(-1)  # shape [B]
        pred01 = torch.sigmoid(raw)  # in [0,1]
        pred = pred01 * 4.0 + 1.0  # bound to [1,5]
        outputs = {"logits": pred.unsqueeze(-1)}  # Trainer expects "logits"
        if labels is not None:
            labels = labels.to(pred.dtype)
            loss = F.mse_loss(pred, labels)
            outputs["loss"] = loss
        return outputs


def tokenize_function(examples):
    out = reward_tokenizer(
        examples["kp"],
        examples["history_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    out["user_id"] = examples["user_id"]
    return out


def calculate_rouge_score(row):
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    for rev in row['filtered_hist_vote_written']:
        scores = scorer.score(row['key_point'], rev)
        rouge1_scores += [scores['rouge1'].fmeasure]
        rouge2_scores += [scores['rouge2'].fmeasure]
        rougel_scores += [scores['rougeL'].fmeasure]

    row['rouge1_scores'] = rouge1_scores
    row['rouge2_scores'] = rouge2_scores
    row['rougel_scores'] = rougel_scores

    return row


def filter_hist_vote_written(row):
    my_df = row[['filtered_hist_vote_written', 'rouge1_scores', 'rouge2_scores', 'rougel_scores']].to_frame().T
    my_df = my_df.explode(['filtered_hist_vote_written', 'rouge1_scores', 'rouge2_scores', 'rougel_scores'])
    my_df = my_df.sort_values(by=['rougel_scores', 'rouge2_scores', 'rouge1_scores'], ascending=False)
    row['sorted_hist_vote_written'] = my_df['filtered_hist_vote_written'].tolist()
    return row


def join_history(reviews):
    # Use tokenizer-specific separator; helps the encoder segment reviews
    sep = reward_tokenizer.sep_token if reward_tokenizer.sep_token is not None else " "
    return f" {sep} ".join([r.strip() for r in reviews])


def calculate_persona_reward(df, persona_reward_prompt, persona_reward_model):
    product_name = df.iloc[0]['product_name']
    ext = re.findall(r"# Personalized Summary:\n*((?:.+\n*)+)", df.iloc[0]['generated_personalized_summaries'])
    if len(ext) > 0:
        generated_personalized_summary = ext[0]
    else:
        generated_personalized_summary = df.iloc[0]['generated_personalized_summaries']
    hist_vote_written = [rev.strip() for rev in df.iloc[0]['hist_vote_written']]
    user_profile = df.iloc[0]['user_profiles']

    prompt = persona_reward_prompt % (product_name, generated_personalized_summary, user_profile)

    all_responses = get_scoring_completion(prompt, persona_reward_model)
    rating_extractions = [[int(rating) for rating in re.findall(r'[0-9]+', ext_response)[:6] if 1 <= int(rating) <= 5]
                          for ext_response in all_responses]
    rating_extractions = [run for run in rating_extractions if len(run) == 6]

    if len(rating_extractions) == 0:
        personal_utterance_reward = 1
    else:
        personal_utterance_reward = statistics.mean([statistics.mean(run) for run in rating_extractions])

    return personal_utterance_reward, rating_extractions, all_responses


def calculate_kp_helpfulness_score(train_df, reward_model, entry_id, generated_summary, generated_summary_kps):
    kp_df = pd.DataFrame(
        {'id': [entry_id for i in range(len(generated_summary_kps))], 'key_point': generated_summary_kps})
    kp_df['generated_personalized_summaries'] = generated_summary
    kp_df = kp_df.merge(train_df[['id', 'product_name', 'category', 'user_id']])

    kp_df = kp_df.merge(train_df)
    kp_df = kp_df.apply(calculate_rouge_score, axis=1)
    kp_df = kp_df.apply(filter_hist_vote_written, axis=1)
    kp_df = kp_df[
        ['id', 'category', 'product_name', 'user_id', 'user_profiles', 'key_point', 'sorted_hist_vote_written']]. \
        rename(columns={'sorted_hist_vote_written': 'hist_vote_written'})
    kp_df['history_text'] = kp_df['hist_vote_written'].apply(join_history)
    kp_df = kp_df.rename(columns={'key_point': 'kp', 'user_id': 'user_id'})

    kp_data = Dataset.from_pandas(kp_df)
    tokenized_kp_data = kp_data.map(tokenize_function, batched=True, remove_columns=kp_data.column_names)
    tokenized_kp_data.set_format("torch")
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(tokenized_kp_data, batch_size=2)

    reward_model.eval()
    output = []
    for reward_batch in eval_dataloader:
        #     batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        batch_inputs = reward_batch['input_ids'].to("cpu")
        batch_masks = reward_batch['attention_mask'].to("cpu")
        with torch.no_grad():
            output += reward_model(batch_inputs, batch_masks)["logits"].view(1, -1).tolist()[0]

    kp_df['predicted_kp_helpfulness'] = output

    return kp_df


def save_rl_data(root_path, kp_df):
    os.makedirs(root_path, exist_ok=True)
    category = kp_df['category'].iloc[0]
    user_id = kp_df['user_id'].iloc[0]
    idx = str(kp_df['id'].iloc[0])
    kp_df.to_pickle(root_path + f"/{category}_{idx}_{user_id}.pkl")


reward_tokenizer = None
MAX_LEN = 512


def train():
    # Read RL Data Subset
    train_df = pd.read_pickle("../data/rl_train/train.pkl")
    train_df = train_df.reset_index().rename(columns={'index': 'id'})
    train_df['user_profiles'] = train_df['s3_user_profiles']
    train_data = train_df

    TEMPLATE = get_prompt("helpfulsumm_cot_helpful_pos")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    base_model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    source_path = '../models/stage_1_helpfulsumm_ft/checkpoint-800'
    output_dir = f'../models/stage_2_helpfulsumm_rl_2/'
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed=seed)

    policy_model = AutoModelForCausalLM.from_pretrained(
        source_path,
        device_map='auto',
        trust_remote_code=False,
        revision="main"
    )
    policy_model = exllama_set_max_input_length(policy_model, max_input_length=65536)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ref_model = create_reference_model(policy_model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model)
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )
    tokenizer.add_prefix_space = False

    print('convert data ...')
    # generate_and_tokenize_prompt_gen = make_generate_and_tokenize_prompt_gen(TEMPLATE, tokenizer)

    def generate_and_tokenize_prompt_gen(data_point):
        return generate_and_tokenize_prompt(TEMPLATE, tokenizer, data_point, gen=True)
    train_data = Dataset.from_pandas(train_data).map(generate_and_tokenize_prompt_gen)
    train_data = train_data.select_columns(['input_ids', 'attention_mask', 'labels', 'id'])

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        ref_model.is_parallelizable = True
        ref_model.model_parallel = True
        policy_model.is_parallelizable = True
        policy_model.model_parallel = True

    micro_batch_size = 1
    _batch_size = micro_batch_size
    gradient_accumulation_steps = 1
    _batch_size = micro_batch_size
    _lr = learning_rate
    config = PPOConfig(
        reward_model=None,
        kl_penalty="kl",
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ppo_epochs=num_epochs,
        learning_rate=_lr,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = PPOTrainer(
        config=config,
        tokenizer=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        dataset=train_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1000,
        #     "max_new_tokens": 15,
        "eos_token_id": tokenizer.eos_token_id,
    }

    api_key = ""
    client = OpenAI(
        api_key=api_key
    )
    # Helpful Opinion Reward Model
    model = "gpt-4.1"
    base_prompt = get_prompt("summary_kp_extraction")
    reward_model_checkpoint = "microsoft/deberta-v2-xlarge"  # swap to "bert-base-uncased" if you prefer BERT
    reward_model = CrossEncoderRegressor(reward_model_checkpoint)
    reward_model.load_state_dict(torch.load('../models/stage_2_helpful_opinion_reward_deberta_ft/model_3_epoch_good.pth',
                                            map_location=torch.device("cpu")))
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_checkpoint, use_fast=True)

    # Persona Consistency Reward Model
    persona_reward_model = "gpt-3.5-turbo"
    persona_reward_prompt = get_prompt("persona_alignment_reward_scoring")

    # Train
    disable_progress_bar()
    epochs = num_epochs
    print("TOTAL EPOCHS: ", num_epochs)
    step_i = 0
    root_path = output_dir + "rl_output/"

    responses = []
    for epoch in tqdm(range(epochs), "epoch:", ncols=100):
        for batch in tqdm(trainer.dataloader, desc=f"Batch (Epoch {epoch + 1})", leave=False, ncols=100):
            query_tensors = batch["input_ids"]
            input_tensors_b = [process_query_tensor(qt) for qt in query_tensors]
            response_tensors_b = []
            for input_tensors in input_tensors_b:
                response_tensors = trainer.generate([input_tensors], return_prompt=False, **generation_kwargs)
                response_tensors_b += response_tensors

            response_b = [tokenizer.decode(rt, skip_special_tokens=True).strip() for rt in response_tensors_b]
            responses += response_b
            batch['response'] = response_b
            id_b = batch['id'].tolist()

            print(f'step-{step_i} >>>')

            reward_b = []
            for summ in response_b:
                if len(summ.strip()) > 0:
                    # KP HELPFULNESS REWARD
                    generated_summary_kps = claim_extraction_from_summary(base_prompt, summ)
                    if len(generated_summary_kps) > 0:
                        kp_df = calculate_kp_helpfulness_score(train_df, reward_model, id_b[0], summ, generated_summary_kps)
                        kp_helpfulness_reward = kp_df['predicted_kp_helpfulness'].mean()
                        kp_df['generated_personalized_summaries'] = summ

                        # PERSONA CONSISTENCY REWARD
                        personal_utterance_reward, rating_extractions, all_responses = calculate_persona_reward(kp_df, persona_reward_prompt, persona_reward_model)

                        # SAVE
                        kp_df['kp_helpfulness_reward'] = kp_helpfulness_reward
                        kp_df['personal_utterance_rating_extractions'] = [rating_extractions for i in range(len(kp_df))]
                        kp_df['personal_utterance_all_responses'] = [all_responses for i in range(len(kp_df))]
                        kp_df['personal_utterance_reward'] = personal_utterance_reward
                        save_rl_data(root_path, kp_df)

                    else:
                        print("############################ NOTE ############################")
                        kp_df = 123
                        kp_helpfulness_reward = 0
                        personal_utterance_reward, rating_extractions, all_responses = calculate_persona_reward(kp_df, persona_reward_prompt, persona_reward_model)

                    # COMBINE
                    final_reward = kp_helpfulness_reward * 0.5 + personal_utterance_reward * 0.5
                    reward_b += [torch.tensor(final_reward)]
                else:
                    reward_b += [torch.tensor(0)]

            stats = trainer.step([qt for qt in query_tensors], [rt for rt in response_tensors_b], reward_b)

            step_logs = get_step_logs(stats, batch, reward_b, step_i)
            append_jsonl(step_logs, os.path.join(output_dir, 'logs.jsonl'))

            if step_i % 50 == 0:
                checkpoint_folder_name = f"step-{step_i}"
                checkpoint_dir = os.path.join(output_dir, checkpoint_folder_name)
                os.makedirs(checkpoint_dir)
                trainer.model.save_pretrained(checkpoint_dir, safe_serialization=False)
                step_stats = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}
                write_json(step_stats, os.path.join(checkpoint_dir, 'reward_stats.json'))

            step_i += 1
            del query_tensors, input_tensors_b, response_tensors_b, response_tensors
            del kp_helpfulness_reward
            del personal_utterance_reward, rating_extractions, all_responses
            del reward_b, kp_df
            gc.collect()
            torch.cuda.empty_cache()  # harmless; can help reduce fragmentation

        if step_i > 150000:
            break
