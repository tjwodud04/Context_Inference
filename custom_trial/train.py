import os
import torch

from datasets import Dataset
from data import CustomDataset, DataCollatorForSupervisedDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


def train(model, tokenizer, train_dataset, valid_dataset, data_collator, peft_config, adapter_output_dir, new_pretrained_model_dir) :
    # training_arguments = TrainingArguments(
    # output_dir=adapter_output_dir,
    # overwrite_output_dir=True,
    # num_train_epochs=1,
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=1,
    # optim="paged_adamw_32bit",
    # save_steps=25,
    # logging_steps=25,
    # learning_rate=2e-4,
    # weight_decay=0.001,
    # fp16=False,
    # bf16=False,
    # max_grad_norm=0.3,
    # max_steps=-1,
    # warmup_ratio=0.03,
    # group_by_length=True,
    # lr_scheduler_type="constant",
    # max_seq_length=1024,
    # packing = True,
    # seed = 42,
    # report_to=None
    # )

    training_args = SFTConfig(
            output_dir=adapter_output_dir,
            overwrite_output_dir=True,
            # do_train=True,
            # do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=3e-4,
            weight_decay=0.1,
            num_train_epochs=10,
            max_steps=-1,
            lr_scheduler_type="cosine",
            # warmup_steps=args.warmup_steps,
            warmup_steps = 2000,
            # warmup_ratio=0.03,
            log_level="info",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=5,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=2048,
            packing=True,
            seed=42,
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        max_seq_length=None,
        # dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        # packing= False,
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     data_collator=data_collator,
    #     args=training_args,
    # )

    trainer.train()

    trainer.model.save_pretrained(new_pretrained_model_dir)