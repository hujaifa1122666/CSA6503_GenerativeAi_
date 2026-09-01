# Install: pip install transformers torch
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of education is"

result = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    pad_token_id=50256
)

print("Prompt:", prompt)
print("\nGenerated Text:")
print(result[0]["generated_text"])
