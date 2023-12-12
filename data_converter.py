#!/usr/bin/env python

from datasets import load_dataset
from bs4 import BeautifulSoup
import re
import pandas as pd

def cleanup_comment(comment):
    text = BeautifulSoup(comment).get_text(separator = '\n', strip = True)
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    # text = re.sub(r'>>[0-9]+', '', text, flags=re.MULTILINE)
    return text.strip()

def process_thread(examples):
    comments = []
    for thread in examples['text']:
        lines = [line for line in thread.split('\n') if line]
        topic = lines[0].split('<>')[4]
        topic = BeautifulSoup(topic).get_text(strip = True)
        topic = re.sub(r' ?\[.+$', '', topic)
        topic = re.sub(r'★', '', topic)
        topic = re.sub(r'[0-9]+ ?$', '', topic)
        topic = re.sub(r'[０１２３４５６７８９]+ ?$', '', topic)
        comments += [(topic, cleanup_comment(line.split('<>')[3])) for line in lines][1:999]
    return {"topic": [topic for (topic, _) in comments], "comment": [comment for (_, comment) in comments]}

dataset = load_dataset("text", data_dir="scraped", sample_by='document')['train'].train_test_split(test_size=0.1)
dataset = dataset.map(process_thread, batched=True, remove_columns=['text'], batch_size=1, num_proc=16)
dataset = dataset.filter(lambda example: ">>" not in example["comment"])
dataset = dataset.filter(lambda example: len(example["comment"]) < 50)
dataset = dataset.shuffle()
dataset.save_to_disk("converted")