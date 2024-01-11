#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ["WANDB_PROJECT"] = "llmchan_gpt2s"
os.environ["WANDB_LOG_MODEL"] = "false"

USE_DETAILED_TOPIC_PROMPT = True

run_name = "train4000k-pretrained-detailed-topic"
dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test'].select(range(2000))

config = AutoConfig.from_pretrained("rinna/japanese-gpt2-small")
# desired_dropout_rate = 0.5
# config.resid_pdrop = desired_dropout_rate
# config.attn_pdrop = desired_dropout_rate
# config.embd_pdrop = desired_dropout_rate
# model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small", config=config, device_map={"": 0})
model = AutoModelForCausalLM.from_pretrained("lm_finetuning-train4000k", config=config, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
tokenizer.do_lower_case = True

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['comment'])):
        if USE_DETAILED_TOPIC_PROMPT:
            detailed_topic = example['detailed_topic'][i].split('\n')
            text = ""
            for j in range(len(detailed_topic), -1, -1):
                shorter_topic = '\n'.join(detailed_topic[0:j])
                text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {shorter_topic}\nコメント: {example['comment'][i]}"
                if len(tokenizer.encode(text)) <= 512:
                    break
        else:
            text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {example['topic'][i]}\nコメント: {example['comment'][i]}"
        output_texts.append(text)
    return output_texts

collator = DataCollatorForCompletionOnlyLM("コメント: ", tokenizer=tokenizer)


if USE_DETAILED_TOPIC_PROMPT:
    batch_size = 16
    acc_step = 4
else:
    batch_size = 64
    acc_step = 1

training_params = TrainingArguments(
    output_dir="./" + run_name,
    run_name=run_name,
    num_train_epochs=10,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=acc_step,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=300)],
)

trainer.train(resume_from_checkpoint=True)
# trainer.train()

trainer.save_model()
