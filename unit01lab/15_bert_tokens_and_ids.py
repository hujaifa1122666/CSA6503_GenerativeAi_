# Install: pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "I love artificial intelligence."
encoded = tokenizer(sentence)

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
token_ids = encoded["input_ids"]

print("Sentence:", sentence)
print("Generated Tokens:", tokens)
print("Token IDs:", token_ids)
