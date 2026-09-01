# Install: pip install transformers torch
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence = "The bank approved my loan."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

print("Sentence:", sentence)
print("Embedding Shape:", embeddings.shape)
print("First 10 values of CLS embedding:")
print(embeddings[0, 0, :10])
