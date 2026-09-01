# Customer Support Knowledge-Base Bot

A RAG Document QA system over 16 customer-support help-center articles.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Add a Gemini API key to `.env` for generated answers. If no key is configured,
the application still shows grounded retrieval results so the RAG pipeline can be tested.

## Required demonstrations

1. Account locked troubleshooting.
2. Payment failed troubleshooting.
3. Combined query using Reset Your Password + Change Your Email Address.
4. Optional out-of-scope query to demonstrate refusal.
