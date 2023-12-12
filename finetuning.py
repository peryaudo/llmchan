#!/usr/bin/env python
# coding: utf-8

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd

dataset = load_from_disk("converted")
dataset = dataset.map(lambda example: {"text": "ニュース:" + example["topic"] + "\n5chの反応:" + example["comment"]}, remove_columns=["topic", "comment"])
train_dataset = dataset['train']
eval_dataset = dataset['test']

train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))
eval_dataset = eval_dataset.select(range(int(len(eval_dataset) * 0.01)))

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "rinna/youri-7b",
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    eval_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

# TODO: >>1 tends to be too long for the context length. Is it possible to truncate it beforehand?
# TODO: Check if the fine tuned weight has better performance than youri-7b-instruction
# TODO: Pretrain LLM with 5ch style posts beforehand (like ULMFiT)