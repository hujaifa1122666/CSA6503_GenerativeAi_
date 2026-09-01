# Install: pip install transformers
from transformers import BertTokenizer, GPT2Tokenizer

sentence = "Transformers are amazing!"

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

bert_tokens = bert_tokenizer.tokenize(sentence)
gpt2_tokens = gpt2_tokenizer.tokenize(sentence)

print("Sentence:", sentence)

print("\nBERT Tokens:")
print(bert_tokens)
print("BERT Token IDs:")
print(bert_tokenizer.encode(sentence))

print("\nGPT-2 Tokens:")
print(gpt2_tokens)
print("GPT-2 Token IDs:")
print(gpt2_tokenizer.encode(sentence))
