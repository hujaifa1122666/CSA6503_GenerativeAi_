# Q40 – Intelligent AI Assistant

## Assignment
Develop a complete AI application integrating chatbot, text summarization, and machine translation capabilities with a polished user interface.

## Included Tasks
1. Chatbot
2. Text summarizer
3. English ↔ Tamil translator

## AI Techniques
### Chatbot
Semantic retrieval using `all-MiniLM-L6-v2`.

### Summarizer
Extractive TextRank-style summarization. Sentence embeddings create a semantic similarity graph and central sentences are selected while preserving original order.

### Translator
Local MarianMT models:
- `Helsinki-NLP/opus-mt-en-ta`
- `Helsinki-NLP/opus-mt-ta-en`

No API key is required.

## Run
Because PyTorch and transformer model files can be large, use a short virtual-environment path on Windows:

```powershell
py -m venv C:\venvs\q40assistant
C:\venvs\q40assistant\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\venvs\q40assistant\Scripts\Activate.ps1
```

The first use of the chatbot/summarizer downloads the sentence-embedding model. The first use of translation downloads the selected MarianMT model.

## UI
The sidebar lets the user select:
- Chatbot
- Summarizer
- Translator

## Invalid Input Handling
The application checks for:
- Empty input
- Very short input
- Same source and target language
- Translation model/download errors

## Test Cases
See `test_cases.json`.

## Architecture
Streamlit UI → Task Selection → Task-specific processing:
- Chatbot → Embedding → Similarity Search → Response
- Summarizer → Sentence Segmentation → Embeddings → Similarity Graph → Ranking → Summary
- Translator → MarianMT → Translated Text

## Limitations
- Chatbot knowledge is limited to the bundled academic FAQ data.
- Extractive summarization does not generate new wording.
- Translation quality can vary for complex or domain-specific text.
- Local transformer models require RAM/disk and an internet connection on first download.

## Possible Improvements
- Add RAG over college PDFs.
- Add a generative LLM for natural chatbot responses.
- Add abstractive summarization.
- Add more Indian languages.
- Add model quantization/caching.
- Add user authentication and analytics.
