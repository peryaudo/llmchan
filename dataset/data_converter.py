#!/usr/bin/env python

from datasets import load_dataset, DatasetDict
from bs4 import BeautifulSoup
import re

_NUM_PROC = 16

def cleanup_topic(topic):
    topic = BeautifulSoup(topic).get_text(strip = True)
    topic = re.sub(r'★.+$', '', topic.replace('[無断転載禁止]', '').replace('©2ch.net', '').strip())
    return topic.encode("utf-8", "ignore").decode("utf-8")

def cleanup_comment(comment):
    comment = BeautifulSoup(comment).get_text(separator = '\n', strip = True).encode("utf-8", "ignore").decode("utf-8")
    return '\n'.join([line for line in comment.split('\n') if not re.match(r'^[＞>]', line)])

def cleanup_detailed_topic(comment):
    detailed_topic = BeautifulSoup(comment).get_text(separator = '\n', strip = True).encode("utf-8", "ignore").decode("utf-8")
    detailed_topic = '\n'.join([line for line in detailed_topic.split('\n') if 'ttp' not in line])
    return detailed_topic

def to_dict_of_lists(l):
    results = {}
    for d in l:
        for k, v in d.items():
            if k in results:
                results[k].append(v)
            else:
                results[k] = []
    return results

def process_thread(examples):
    comments = []
    for thread in examples['text']:
        lines = [line for line in thread.split('\n') if line]
        topic = cleanup_topic(lines[0].split('<>')[4])
        detailed_topic = cleanup_detailed_topic(lines[0].split('<>')[3])
        for index, line in enumerate(lines):
            if index == 0:
                continue
            if "Over 1000 Thread" in line.split('<>')[2]:
                continue
            comments.append(
                {"index": index + 1,
                 "topic": topic,
                 "detailed_topic": detailed_topic,
                 "comment": cleanup_comment(line.split('<>')[3])})
    return to_dict_of_lists(comments)

def filter_example(example):
    return (len(example["comment"]) < 100) and (">>" not in example["comment"]) and ("ttp" not in example["comment"]) and ("ID" not in example["comment"]) and ("批判要望" not in example["topic"]) and ("◆" not in example["topic"])

train_dataset = load_dataset("text", data_dir="scraped", sample_by='document', split='train')
test_dataset = load_dataset("text", data_dir="scraped_val", sample_by='document', split='train')
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
dataset = dataset.map(process_thread, batched=True, remove_columns=['text'], batch_size=1, num_proc=_NUM_PROC)
dataset = dataset.filter(filter_example)
dataset = dataset.shuffle()
dataset.save_to_disk("converted")