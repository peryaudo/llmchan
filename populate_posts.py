import random
import sqlite3
from tqdm import tqdm
from transformers import pipeline
import torch
pipe = pipeline(task="text-generation", model="rinna/youri-7b-chat", device=0, torch_dtype=torch.float16)

fewshot_examples = [tuple(uttr.split(': ')) for uttr in open('fewshot_examples.txt').readlines()]

def generate(topic):
    instruction = "匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。"
    input = topic
    
    context = [
        {
            "speaker": "設定",
            "text": instruction
        },
        {
            "speaker": "ユーザー",
            "text": input
        }
    ]
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in context
    ]
    prompt = "\n".join(prompt)
    prompt = (
        prompt
        + "\n"
        + "システム: "
    )

    generated_text = pipe(prompt, max_new_tokens=500, do_sample=True)[0]['generated_text']
    return generated_text[len(prompt):]

conn = sqlite3.connect('database.db')

for thread_id, title in tqdm(conn.execute("SELECT id, topic FROM thread").fetchall()):
    for _ in range(random.randint(1, 5)):
        generated_text = generate(title)
        conn.execute("INSERT INTO post (thread_id, body) VALUES (?, ?)", (thread_id,generated_text))
        conn.commit()

conn.close()