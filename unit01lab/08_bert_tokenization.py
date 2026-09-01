# Install: pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "Artificial intelligence is changing the world."
tokens = tokenizer.tokenize(sentence)

print("Sentence:", sentence)
print("BERT Tokens:", tokens)
