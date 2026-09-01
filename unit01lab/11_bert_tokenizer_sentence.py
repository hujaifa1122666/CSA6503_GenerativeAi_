# Install: pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "Machine learning is useful."
tokens = tokenizer.tokenize(sentence)

print("Sentence:", sentence)
print("Tokens:", tokens)
