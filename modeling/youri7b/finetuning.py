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
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel, prepare_model_for_kbit_training
from peft.utils import prepare_model_for_kbit_training
import torch

os.environ["WANDB_PROJECT"] = "llmchan_youri7b"
os.environ["WANDB_LOG_MODEL"] = "false"

USE_DETAILED_TOPIC_PROMPT = False

run_name = "train4000k-pretrained"
dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test'].select(range(2000))

BASE_MODEL_NAME = "rinna/youri-7b"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map={"": 0}, quantization_config=quant_config)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model = prepare_model_for_kbit_training(base_model)
model = PeftModel.from_pretrained(base_model, "lm_finetuning-train4000k", is_trainable=True, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

response_template = "\nコメント:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


if USE_DETAILED_TOPIC_PROMPT:
    batch_size = 16
    acc_step = 4
else:
    batch_size = 32
    acc_step = 1

training_params = TrainingArguments(
    output_dir="./" + run_name,
    run_name=run_name,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=acc_step,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_steps=200,
    logging_steps=100,
    eval_steps=200,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="wandb",
    load_best_model_at_end=True,
    save_total_limit=10,
    save_safetensors=False,
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
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

trainer.save_model()
