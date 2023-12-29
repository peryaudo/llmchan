#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ["WANDB_PROJECT"] = "llmchan_gpt2s"
os.environ["WANDB_LOG_MODEL"] = "false"

dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test']

train_dataset = train_dataset.select(range(10000))
eval_dataset = eval_dataset.select(range(1000))

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small", device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
tokenizer.do_lower_case = True

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['comment'])):
        text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {example['topic'][i]}\nコメント: {example['comment'][i]}"
        output_texts.append(text)
    return output_texts

collator = DataCollatorForCompletionOnlyLM("コメント: ", tokenizer=tokenizer)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    # gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_steps=200,
    logging_steps=100,
    eval_steps=200,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
