# Install: pip install transformers torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Once upon a time in a small village"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("Prompt:", prompt)
print("\nGenerated Text:")
print(generated_text)
