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
import numpy as np
from trl import SFTTrainer

os.environ["WANDB_PROJECT"] = "llmchan_gpt2s"
os.environ["WANDB_LOG_MODEL"] = "false"

run_name = "lm_finetuning-train4000k"
dataset = load_from_disk("../../converted")
train_dataset = dataset['train']
eval_dataset = dataset['test'].select(range(2000))

config = AutoConfig.from_pretrained("rinna/japanese-gpt2-small")
# desired_dropout_rate = 0.5
# config.resid_pdrop = desired_dropout_rate
# config.attn_pdrop = desired_dropout_rate
# config.embd_pdrop = desired_dropout_rate
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small", config=config, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
tokenizer.do_lower_case = True

training_params = TrainingArguments(
    output_dir="./" + run_name,
    run_name=run_name,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_steps=100,
    logging_steps=25,
    eval_steps=100,
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
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
