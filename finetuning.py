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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd

dataset = load_from_disk("converted")
train_dataset = dataset['train']
eval_dataset = dataset['test']

# train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.1)))
eval_dataset = eval_dataset.select(range(int(len(eval_dataset) * 0.1)))

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "rinna/youri-7b-chat",
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['comment'])):
        text = f"設定: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nユーザー: {example['topic'][i]}\nシステム: {example['comment'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "\nシステム:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

peft_args = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

# TODOs:
# - Try larger gradient accumulation to mitigate oscilating loss?
# - Try gradient checkpointing?
# - Enable flash attention 2?
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=50,
    eval_steps=500,
    learning_rate=1e-4,
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
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

trainer.save_model()
