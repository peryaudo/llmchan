#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig,
    EarlyStoppingCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel

os.environ["WANDB_PROJECT"] = "llmchan_gpt2s"
os.environ["WANDB_LOG_MODEL"] = "false"

run_name = "train4000k-pretrained-lora256-1e-4"
dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test'].select(range(2000))

config = AutoConfig.from_pretrained("rinna/japanese-gpt2-small")
# desired_dropout_rate = 0.5
# config.resid_pdrop = desired_dropout_rate
# config.attn_pdrop = desired_dropout_rate
# config.embd_pdrop = desired_dropout_rate
base_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small", config=config, device_map={"": 0})
model = PeftModel.from_pretrained(base_model, "lm_finetuning-train4000k-lora256-all", device_map={"": 0})
for name, param in model.named_parameters():
    if 'lora' in name or 'Lora' in name:
        param.requires_grad = True
# model = model.merge_and_unload()
# model.save_pretrained("lm_finetuning-train4000k-lora256-all")

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
    output_dir="./" + run_name,
    run_name=run_name,
    num_train_epochs=10,
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
    report_to="wandb",
    load_best_model_at_end=True,
    save_total_limit=20,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_num_proc=32,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=150)],
)

trainer.train()

trainer.save_model()
