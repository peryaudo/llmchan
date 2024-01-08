from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_from_disk
import torch

model_name = "train4000k-pretrained/checkpoint-58200"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
tokenizer.do_lower_case = True

def generate(topic):
    text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {topic}\nコメント: "
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=200,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output = tokenizer.decode(output_ids.tolist()[0])[len(tokenizer.decode(token_ids.tolist()[0])):]
    print(output)

def generate_n(topic, n):
    for i in range(n):
        generate(topic)

dataset = load_from_disk("../../converted")["test"].select(range(10))

print(f"model: {model_name}")
print("=" * 50)
for example in dataset:
    print(example["topic"])
    print("=" * 50)
    generate_n(example["topic"], 20)
    print("")