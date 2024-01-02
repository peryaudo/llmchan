from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
import torch.nn.functional as F

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def add_similarity_score(example):
    topic_embedding = model.encode(example['topic'], convert_to_tensor=True)
    comment_embedding = model.encode(example['comment'], convert_to_tensor=True)
    return {"similarity": F.cosine_similarity(topic_embedding, comment_embedding)}

dataset = load_from_disk("converted")
dataset = dataset.map(add_similarity_score, batched=True, batch_size=1000, num_proc=1)
dataset.save_to_disk("converted_with_score")