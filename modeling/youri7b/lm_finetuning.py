#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
import numpy as np
from trl import SFTTrainer
from peft import LoraConfig

os.environ["WANDB_PROJECT"] = "llmchan_youri7b"
os.environ["WANDB_LOG_MODEL"] = "false"

run_name = "lm_finetuning-train4000k"
dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test'].select(range(2000))

BASE_MODEL_NAME = "rinna/youri-7b"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map={"": 0}, quantization_config=quant_config)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.0,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
)

training_params = TrainingArguments(
    output_dir="./" + run_name,
    run_name=run_name,
    num_train_epochs=10,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_steps=100,
    logging_steps=25,
    eval_steps=100,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=False,
    report_to="wandb",
    load_best_model_at_end=True,
    save_total_limit=20,
    save_safetensors=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_args,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=True,
    dataset_text_field='comment',
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

trainer.save_model()
