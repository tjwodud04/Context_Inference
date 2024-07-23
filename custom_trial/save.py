import os
import torch

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from data import CustomDataset, DataCollatorForSupervisedDataset
from trl import SFTTrainer


def merge_and_save(base_model_path, name, adapter_output_dir, new_merged_model_path) :
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                # load_in_4bit=True,
                # quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
        )

    # base_model.config.use_cache = False # silence the warnings
    # base_model.config.pretraining_tp = 1
    # base_model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    # tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_eos_token = True
    # tokenizer.add_bos_token, tokenizer.add_eos_token


    model = PeftModel.from_pretrained(base_model, adapter_output_dir)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    model.save_pretrained(new_merged_model_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(new_merged_model_path)
    print(f"Model saved to {new_merged_model_path}")

if __name__ == "main" :
    merge_and_save(
        base_model_path = "/home/work/jysuh/Trial/Third_trial/base_model/llama-3-Korean-Bllossom-8B/",
        name = "Bllossom",
        adapter_output_dir = f"/home/work/jysuh/Trial/Third_trial/results/{name}/Adapter/checkpoint-950/",
        new_merged_model_path = f"/home/work/jysuh/Trial/Third_trial/results/{name}/final/"
        )
