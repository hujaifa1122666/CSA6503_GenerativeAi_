# Q2 Assessment Report – Student Academic Support Chatbot

## 1. Problem Statement
Develop an AI chatbot that assists students with questions related to subjects, assignments, examinations, attendance, and academic schedules.

## 2. Objective
Build a user-friendly academic chatbot using semantic retrieval.

## 3. AI Technique
`all-MiniLM-L6-v2` sentence embeddings + cosine similarity over an academic FAQ knowledge base.

## 4. Application Architecture
User Interface → Validation → Embedding Model → Similarity Search → FAQ Answer → Response Display.

## 5. Features
- Subject queries
- Assignment queries
- Examination queries
- Attendance queries
- Academic schedule queries
- Fallback response
- Empty/invalid input handling
- Similarity score display

## 6. Testing
Test normal, paraphrased, unrelated, empty, and short inputs using `test_cases.txt`.

## 7. Limitations
The chatbot depends on the provided knowledge base and does not use live institutional data.

## 8. Future Improvements
Add college documents, RAG, authentication, live timetable integration, and an LLM response layer.
