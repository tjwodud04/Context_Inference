import os
import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from data import CustomDataset, DataCollatorForSupervisedDataset
from trl import SFTTrainer, SFTConfig

def load_model(base_model_path) :
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            # load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )

    model.config.use_cache = False # silence the warnings
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, pad_to_max_length=True)
    # tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_eos_token = True
    # tokenizer.add_bos_token, tokenizer.add_eos_token

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )

    model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config