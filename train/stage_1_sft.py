from dataclasses import dataclass, field
import pathlib
import typing
import os
import re
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from auto_gptq import exllama_set_max_input_length

# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def format_profile(user_profiles):
    profile_lines = re.findall(r"\n[^\:]+:([^\n]+)", user_profiles)
    profile_lines = [line.strip("[*\n \.]") for line in profile_lines]
    profile_lines = [line for line in profile_lines if len(line) > 1]
    return ". ".join(profile_lines) + "."


def make_format_chat(tokenizer, base_prompt, output_template):
    def format_chat(row):
        texts = []
        for i in range(len(row['product_name'])):
            # Use the model’s native chat template for best results
            product_name = row['product_name'][i]
            import random
            random.seed(42)
            hist_vote_written = random.sample(row['filtered_hist_vote_written'][i], min(5, len(row['filtered_hist_vote_written'][i])))
            product_reviews = row['product_reviews'][i]

            if len(product_reviews) > 5:
                import random
                random.seed(42)
                product_reviews = random.sample(product_reviews, 5)

            input_text = base_prompt.format(product_name=product_name, product_reviews=product_reviews,
                                            hist_vote_written=hist_vote_written)
            output_text = output_template.format(user_profiles=format_profile(row['s3_user_profiles'][i]),
                                                 helpful_kps="\n".join(
                                                     ["- " + kp for i, kp in enumerate(row['s4_helpful_kps_filtered'][i])]),
                                                 personalized_summaries=row['s5_annotated_personalized_summaries'][i])

            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            # include assistant text (SFT target). No generation prompt for SFT.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts
    return format_chat

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # if lora_args.q_lora:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    #     if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
    #         logging.warning(
    #             "FSDP and ZeRO3 are both currently incompatible with QLoRA."
    #         )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    quantization_config_loading = transformers.GPTQConfig(bits=4, disable_exllama=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config_loading,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=[
            "k_proj",
            "o_proj",
            "q_proj",
            "v_proj"
        ],
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)

    model.print_trainable_parameters()

    # model.requires_grad_(False)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
    )
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token = "</s>"
    tokenizer.add_prefix_space = False

    data_path = data_args.data_path
    training_args.fsdp_config = {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}

    df = pd.read_pickle(data_path)
    train = Dataset.from_pandas(df)

    base_prompt = get_prompt("helpfulsumm_cot_helpful_pos")
    output_template = """# User Profile:
{user_profiles}
# Helpful Key Points:
{helpful_kps}
# Personalized Summary:
{personalized_summaries}"""
    format_chat = make_format_chat(tokenizer, base_prompt, output_template)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=train,
        formatting_func=format_chat,
        # dataset_text_field='train_text',
        max_seq_length=training_args.model_max_length,
        peft_config=lora_config
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
