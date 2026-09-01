# Install: pip install transformers torch
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence = "Deep learning helps computers understand patterns."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

sentence_embedding = outputs.last_hidden_state[:, 0, :]

print("Sentence:", sentence)
print("Embedding Shape:", sentence_embedding.shape)
print("First 10 values:")
print(sentence_embedding[0, :10])
