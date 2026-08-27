# Q2 – Student Academic Support Chatbot

## Assignment
Develop a chatbot that assists students with questions related to subjects, assignments, examinations, attendance, and academic schedules.

## Technique
This application uses **semantic retrieval** with `sentence-transformers/all-MiniLM-L6-v2`.
The user query is converted to an embedding and compared against the embeddings of a curated academic FAQ knowledge base. The answer belonging to the most similar FAQ is returned.

This is a lightweight retrieval-based AI chatbot and does not require an external API key.

## UI
The application is built with Streamlit.

## Run
```powershell
py -m venv C:\venvs\q2chatbot
C:\venvs\q2chatbot\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\venvs\q2chatbot\Scripts\Activate.ps1
```

## Test Cases
Use `test_cases.txt`. The set includes normal questions, paraphrased questions, an unrelated question, empty input, and very short input.

## Architecture
User → Streamlit UI → Input validation → Sentence embedding → Cosine similarity → Academic FAQ retrieval → Response + similarity score.

## Limitations
- Knowledge is limited to the bundled FAQ collection.
- It does not access live college schedules.
- Similarity thresholds may need tuning.
- It does not replace official academic communication.

## Improvements
- Add college-specific documents.
- Add RAG with PDFs/handbooks.
- Add authentication and student-specific timetable data.
- Add an LLM response layer after retrieval.
