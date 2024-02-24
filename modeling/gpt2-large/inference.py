import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_from_disk
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of the model", default="train4000k-pretrained/checkpoint-58200")
parser.add_argument("--detailed_topic", type=bool, help="Use the detailed topic for querying", default=False)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": 0}, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-1b", use_fast=False)
tokenizer.do_lower_case = True

def generate(example):
    if args.detailed_topic:
        detailed_topic = example['detailed_topic'].split('\n')
        text = ""
        for j in range(len(detailed_topic), -1, -1):
            shorter_topic = '\n'.join(detailed_topic[0:j])
            text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {shorter_topic}\nコメント: "
            if len(tokenizer.encode(text)) <= 312:
                break
    else:
        text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {example['topic']}\nコメント: "
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=200,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output = tokenizer.decode(output_ids.tolist()[0])[len(tokenizer.decode(token_ids.tolist()[0])):]
    print(output)

def generate_n(example, n):
    for i in range(n):
        generate(example)

dataset = load_from_disk("../../converted")["test"].select(range(10))

print(f"model: {args.model_name}")
print("=" * 50)
for example in dataset:
    print(example["topic"])
    print("=" * 50)
    generate_n(example, 20)
    print("")
