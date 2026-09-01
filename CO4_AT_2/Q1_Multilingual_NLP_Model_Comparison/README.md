# Q1 – Compare Four Models for Multilingual NLP

## Assignment alignment
Compare BERT, GPT, T5 and LLaMA for multilingual NLP applications, especially English and Indian languages.

## Models used for the runnable demo
- BERT → `bert-base-multilingual-cased`
- GPT → `openai-community/gpt2` as a lightweight GPT-family representative
- T5 → `google/mt5-small`
- LLaMA → `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as a lightweight LLaMA-family representative

## Why representative checkpoints?
Exact production checkpoints can be large or gated. The code therefore uses accessible lightweight checkpoints so the demonstration can run locally without API keys. The presentation/report should clearly distinguish model-family comparison from checkpoint-level performance.

## Run
```powershell
py -m venv C:\venvs\co4at2q1
C:\venvs\co4at2q1\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Demonstration
Use English, Hindi, Tamil and Telugu examples in the UI.

## Architecture summary
BERT: encoder-only → contextual understanding.
GPT: decoder-only → autoregressive generation.
T5/mT5: encoder-decoder → text-to-text tasks.
LLaMA: decoder-only → autoregressive generation/chat.

## Source basis
The official model cards/documentation for FLAN-T5 and mT5 describe text-to-text and multilingual capabilities. The mT5 checkpoint covers more than 100 languages, including several Indian languages. The LLaMA family is decoder-only and autoregressive.
