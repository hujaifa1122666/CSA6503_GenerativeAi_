# Install: pip install transformers torch
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

text = "The product is excellent and I am very happy with it."
result = sentiment_analyzer(text)

print("Text:", text)
print("\nSentiment Analysis Result:")
print(result)
