# Q40 Assessment Report – Intelligent AI Assistant

## 1. Problem Statement
Develop a complete AI application integrating chatbot, text summarization, and machine translation capabilities through a polished UI.

## 2. Objective
Build one application with three selectable AI tasks.

## 3. Architecture
Streamlit UI → Task Selection → AI Processing → Result Display.

### Chatbot
Sentence embeddings + semantic retrieval.

### Summarizer
Extractive TextRank-style sentence ranking using sentence embeddings.

### Translator
Hugging Face MarianMT models for English ↔ Tamil.

## 4. Features
- Chatbot
- Summarization
- Translation
- Task selection
- Empty-input handling
- Short-input validation
- Translation error handling
- Multiple test cases

## 5. Testing
Use `test_cases.json` and demonstrate normal, paraphrased, empty, short, and unexpected inputs.

## 6. Limitations
Discuss knowledge-base coverage, extractive summarization limitations, translation quality, and local model resource requirements.

## 7. Improvements
Add RAG, generative LLM responses, abstractive summarization, more languages, quantization, authentication, and monitoring.
