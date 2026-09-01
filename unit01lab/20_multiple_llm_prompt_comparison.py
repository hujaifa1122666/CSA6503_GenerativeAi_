# Install: pip install transformers torch
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Explain artificial intelligence in simple words:",
    "Write a short paragraph about machine learning:",
    "Describe the future of technology:"
]

for i, prompt in enumerate(prompts, start=1):
    result = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        pad_token_id=50256
    )

    print(f"\n========== PROMPT {i} ==========")
    print("Prompt:", prompt)
    print("Response:")
    print(result[0]["generated_text"])
