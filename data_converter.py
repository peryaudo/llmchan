#!/usr/bin/env python

from datasets import load_dataset
from bs4 import BeautifulSoup
import re
import pandas as pd

def cleanup_topic(topic):
    topic = BeautifulSoup(topic).get_text(strip = True)
    topic = re.sub(r'★.+$', '', topic.replace('[無断転載禁止]', '').replace('©2ch.net', '').strip())
    return topic.encode("utf-8", "ignore").decode("utf-8")

def cleanup_comment(comment):
    return BeautifulSoup(comment).get_text(separator = '\n', strip = True).encode("utf-8", "ignore").decode("utf-8")

def process_thread(examples):
    comments = []
    for thread in examples['text']:
        lines = [line for line in thread.split('\n') if line]
        topic = cleanup_topic(lines[0].split('<>')[4])
        comments += [(index + 1,
                      topic,
                      cleanup_comment(line.split('<>')[3]))
                      for index, line in enumerate(lines)][1:999]

    return {"index": [index for (index, _, _) in comments],
            "topic": [topic for (_, topic, _) in comments],
            "comment": [comment for (_, _, comment) in comments]}

def filter_example(example):
    return (len(example["comment"]) < 100) and (">>" not in example["comment"]) and ("ttp" not in example["comment"]) and ("批判要望" not in example["topic"])

dataset = load_dataset("text", data_dir="scraped", sample_by='document')['train'].train_test_split(test_size=0.01)
dataset = dataset.map(process_thread, batched=True, remove_columns=['text'], batch_size=1, num_proc=16)
dataset = dataset.filter(filter_example)
dataset = dataset.shuffle()
dataset.save_to_disk("converted")