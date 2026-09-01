# Install: pip install transformers torch
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentences = [
    "The cat is sitting on the mat.",
    "A cat is resting on a mat."
]

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state[:, 0, :]

similarity = F.cosine_similarity(
    embeddings[0].unsqueeze(0),
    embeddings[1].unsqueeze(0)
).item()

print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Cosine Similarity:", round(similarity, 4))
