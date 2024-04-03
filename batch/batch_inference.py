import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

model = AutoModelForCausalLM.from_pretrained("./gpt2-large-checkpoint-69200", device_map={"": 0}, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-1b", use_fast=False)
tokenizer.do_lower_case = True

def generate(topic):
    text = f"質問: 匿名掲示板5ちゃんねるの投稿者として、以下のニュースにコメントしてください。\nニュース: {topic}\nコメント: "
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
    return output

print(generate("「裏金」処分、線引き恣意的？　　茂木氏主導、野党も「おかしな話」"))