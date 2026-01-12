import gc
import transformers
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from utils.rl import *

import typing
from dataclasses import dataclass, field
from auto_gptq import exllama_set_max_input_length
from openai import OpenAI
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
import ast
import time
from multiprocessing import Pool
from pathlib import Path
from os import listdir
from tqdm import tqdm
import random
import re

wandb.init(mode="disabled")


def tokenize(tokenizer, prompt):
    result = tokenizer(
        prompt
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(TEMPLATE, tokenizer, data_point, hist_vote_written_size=5):
    product_name = data_point['product_name']
    product_reviews = data_point['product_reviews']
    if len(product_reviews) > 30:
        import random
        random.seed(42)
        product_reviews = random.sample(product_reviews, 30)

    import random
    random.seed(42)
    hist_vote_written = random.sample(data_point['filtered_hist_vote_written'],
                                      min(hist_vote_written_size, len(data_point['filtered_hist_vote_written'])))

    query = TEMPLATE.format(product_name=product_name, product_reviews=product_reviews,
                            hist_vote_written=hist_vote_written)

    msg = [{"role": "user", "content": query}]
    formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    tokenized_full_prompt = tokenize(tokenizer, formatted_prompt)

    return tokenized_full_prompt


def generate_and_tokenize_prompt_gen(data_point, TEMPLATE, tokenizer, hist_vote_written_size):
    return generate_and_tokenize_prompt(TEMPLATE, tokenizer, data_point, hist_vote_written_size)


def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summary_generation(row, hist_vote_written_size, TEMPLATE, tokenizer, trainer, generation_kwargs):
    test_data = pd.DataFrame([row.to_dict()])
    test_data = Dataset.from_pandas(test_data).map(generate_and_tokenize_prompt_gen,
                                                   fn_kwargs={"hist_vote_written_size": hist_vote_written_size,
                                                              "TEMPLATE": TEMPLATE, "tokenizer": tokenizer})
    test_data = test_data.select_columns(['input_ids', 'attention_mask', 'labels'])
    test_data.set_format("torch")
    from torch.utils.data import DataLoader
    test_data = DataLoader(test_data, batch_size=2)

    for batch in test_data:
        query_tensors = batch["input_ids"]
        input_tensors_b = [process_query_tensor(qt) for qt in query_tensors]
        response_tensors_b = []
        for input_tensors in input_tensors_b:
            response_tensors = trainer.generate([input_tensors], return_prompt=False, **generation_kwargs)
            response_tensors_b += response_tensors

        response_b = [tokenizer.decode(rt, skip_special_tokens=True).strip() for rt in response_tensors_b]

    del query_tensors, input_tensors_b, input_tensors, response_tensors, response_tensors_b
    gc.collect()
    torch.cuda.empty_cache()  # harmless; can help reduce fragmentation

    return response_b[0]


def personanlized_summarize(row, TEMPLATE, tokenizer, trainer, generation_kwargs):
    generated_responses = []
    for hist_sample_size in range(5, 35, 5):
        try:
            response = summary_generation(row, hist_sample_size, TEMPLATE, tokenizer, trainer, generation_kwargs)
            generated_responses += [response]

            ext = re.findall(r"Personalized [Ss]ummary(?:\]|[^:\n]*:)[ \n]*((?:.+\n*)+)",
                             response)  # MAY BE CAN TRY THIS
            if len(ext) > 0:
                summary = ext[0]
                return summary, generated_responses
            time.sleep(3)
        except Exception as e:
            print(e)

    return None, generated_responses


def get_claim_split_completion(personalized_summaries, base_prompt, model, client):
    prompt = base_prompt % (personalized_summaries)

    retries = 5
    while retries > 0:
        try:
            response = get_completion(prompt, model, client)
            return response
        except Exception as e:
            if e:
                if "exceeded your current quota" in str(e).lower():
                    raise e
                print(e)
                print('Timeout error, retrying...')
                retries -= 1
                if "limit reached for" in str(e).lower():
                    time.sleep(30)
                else:
                    time.sleep(5)
            else:
                raise e

    print('API is not responding, moving on...')
    return None


def prompted_claim_split_generation(root_path, domain, domain_df, base_prompt, model, client, save_step=10):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    claim_split_predicted_list = []

    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, ": ", "Loaded saved file. Done")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")
        claim_split_predicted_list = last_domain_df['claim_split_predicted'].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        personalized_summaries = row['personalized_summaries']
        claim_split_predicted = get_claim_split_completion(personalized_summaries, base_prompt, model, client)
        claim_split_predicted_list += [claim_split_predicted]
        time.sleep(0.1)

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'claim_split_predicted', claim_split_predicted_list)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'claim_split_predicted', claim_split_predicted_list)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


@dataclass
class InferenceArguments:
    policy_base_model: str
    policy_model_path: str
    inference_data_path: str
    summary_output_dir: str
    kp_extraction_output_dir: str
    openai_api_key: str
    summary_kp_extraction_model: str = field(default="gpt-4.1")
    batch_size: typing.Optional[int] = field(default=1)
    num_epochs: typing.Optional[int] = field(default=1)
    learning_rate: typing.Optional[float] = field(default=2e-7)
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 10
    seed: int = 201
    debug_mode: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def inference():
    parser = transformers.HfArgumentParser(
        (InferenceArguments, LoraArguments)
    )
    (inference_args, lora_args) = parser.parse_args_into_dataclasses()

    merged_df = pd.read_pickle(inference_args.inference_data_path)
    # merged_df = pd.read_pickle("../data/test/test.pkl")
    test_df = merged_df

    TEMPLATE = get_prompt("helpfulsumm_cot_helpful_pos")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    base_model = inference_args.policy_base_model
    source_path = inference_args.policy_model_path
    # base_model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    # source_path = '../models/stage_2_helpfulsumm_rl/step-200/'

    set_seed(seed=inference_args.seed)

    openai_client = OpenAI(
        api_key=inference_args.openai_api_key
    )

    policy_model = AutoModelForCausalLM.from_pretrained(
        source_path,
        device_map='auto',
        trust_remote_code=False,
        revision="main",
        # attn_implementation="flash_attention_2"
    )

    policy_model = exllama_set_max_input_length(policy_model, max_input_length=65536)
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
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
    _lr = inference_args.learning_rate
    config = PPOConfig(
        reward_model=None,
        kl_penalty="kl",
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ppo_epochs=inference_args.num_epochs,
        learning_rate=_lr,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = PPOTrainer(
        config=config,
        tokenizer=tokenizer,
        model=policy_model,
        ref_model=ref_model,
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
        "eos_token_id": tokenizer.eos_token_id,
    }

    for col in ['product_reviews', 'filtered_hist_vote_written']:
        merged_df[col] = merged_df[col].apply(lambda x: [rev.strip() for rev in x])

    # RUN INFERENCE
    test_df = test_df[['category', 'product_name', 'user_id', 'product_reviews', 'filtered_hist_vote_written']]
    test_df['my_category'] = 1
    root_path = inference_args.summary_output_dir
    # root_path = f"../output/stage_2_rl_inference/summary"
    domain = 1
    domain_df = test_df.reset_index(drop=True)
    save_step = 1
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    personalized_summaries = []
    start = 0
    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue
        summary, responses = personanlized_summarize(row, TEMPLATE, tokenizer, trainer, generation_kwargs)
        personalized_summaries += [(summary, responses)]
        time.sleep(3)
        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'personalized_summaries', personalized_summaries)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")
    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'personalized_summaries', personalized_summaries)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")

    # READ GENERATED SUMMARY
    personalized_summary_df = pd.read_pickle(root_path + "/1/1_done.pkl")
    mask = pd.isnull(personalized_summary_df['personalized_summaries'].apply(lambda x: x[0]))
    personalized_summary_df = personalized_summary_df[~mask]
    personalized_summary_df['personalized_summaries'] = personalized_summary_df['personalized_summaries'].apply(lambda x: x[0])
    personalized_summary_df['personalized_summaries'] = personalized_summary_df['personalized_summaries'].apply(
        lambda x: re.sub(r"\n*Note:[^\n]+", "", x).strip("\n").strip())

    # KP EXTRACTION
    model = inference_args.summary_kp_extraction_model
    base_prompt = get_prompt("summary_kp_extraction")
    personalized_summary_df['my_category'] = 1
    root_path = inference_args.kp_extraction_output_dir
    # root_path = f"../output/stage_2_rl_inference/summary_kp_extraction"
    inputs = [(root_path,
               domain,
               personalized_summary_df[personalized_summary_df['my_category'] == domain].reset_index(drop=True),
               base_prompt,
               model,
               openai_client
               )
              for domain in personalized_summary_df['my_category'].unique()]
    num_workers = 1
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompted_claim_split_generation, inputs)
    print("TIME ELAPSED", time.time() - start_time)

    # READ KP EXTRACTION
    claim_summary_df = pd.read_pickle(root_path + "/1/1_done.pkl")
    claim_summary_df['claim_split_predicted'] = claim_summary_df['claim_split_predicted'].apply(lambda x: x.strip("```json").strip("\n"))
    mask = claim_summary_df['claim_split_predicted'].str.contains('Please provide the personalized summary')
    claim_summary_df = claim_summary_df[~mask]
    claim_summary_df['claim_split_predicted'] = claim_summary_df['claim_split_predicted'].apply(lambda x: ast.literal_eval(x))

if __name__ == "__main__":
    inference()