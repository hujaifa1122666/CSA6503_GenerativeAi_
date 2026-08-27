import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
}

FAQS = [
    ("Attendance", "What is the minimum attendance requirement?",
     "Students are expected to maintain the attendance percentage prescribed by the institution. Check the current academic regulations or contact the department office for the exact requirement."),
    ("Assignments", "How can I submit an assignment?",
     "Submit assignments through the platform specified by your faculty and follow the deadline and file-format instructions."),
    ("Examinations", "How do I register for an examination?",
     "Use the official examination-registration portal during the notified registration window."),
    ("Examinations", "Where can I get my hall ticket?",
     "Check the official examination or student portal for the hall-ticket download link."),
    ("Academic Schedule", "Where can I find the academic calendar?",
     "Check the official academic portal or institution website for the latest academic calendar."),
    ("Subjects", "Where can I get my course syllabus?",
     "Download the current syllabus from the official department or student portal."),
    ("Assignments", "What happens if I miss an assignment deadline?",
     "Contact the concerned faculty member promptly. Late-submission rules depend on the course and faculty policy."),
    ("Academic Support", "Who should I contact about an academic issue?",
     "Start with the subject faculty or class advisor and escalate to the department office when necessary."),
]

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@lru_cache(maxsize=1)
def get_faq_index():
    model = get_embedding_model()
    questions = [x[1] for x in FAQS]
    emb = model.encode(questions, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")

def validate_text(text):
    if not text or not text.strip():
        return "Input cannot be empty."
    if len(text.strip()) < 3:
        return "Please enter at least a few meaningful words."
    return None

def chatbot_answer(query, threshold=0.40):
    model = get_embedding_model()
    index = get_faq_index()
    q = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q, index)[0]
    idx = int(np.argmax(scores))
    score = float(scores[idx])
    category, question, answer = FAQS[idx]
    if score < threshold:
        return {
            "answer": "I do not have enough information to answer confidently. Please provide more details or contact the concerned department.",
            "score": score,
            "category": "Fallback",
        }
    return {"answer": answer, "score": score, "category": category}

def split_sentences(text):
    # Handles common English punctuation and keeps reasonably sized sentences.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in pieces if p.strip()]

def summarize_text(text, ratio=0.35):
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text.strip()

    model = get_embedding_model()
    emb = model.encode(sentences, normalize_embeddings=True)
    sim = cosine_similarity(emb, emb)

    # TextRank-style weighted sentence graph.
    scores = np.ones(len(sentences), dtype="float32")
    damping = 0.85
    for _ in range(25):
        new_scores = np.ones(len(sentences), dtype="float32") * (1 - damping)
        for i in range(len(sentences)):
            weights = sim[i].copy()
            weights[i] = 0
            weights = np.maximum(weights, 0)
            total = weights.sum()
            if total == 0:
                continue
            for j in range(len(sentences)):
                if i != j:
                    new_scores[i] += damping * (weights[j] / total) * scores[j]
        scores = new_scores

    count = max(1, min(len(sentences), int(round(len(sentences) * ratio))))
    top_idx = np.argsort(scores)[-count:]
    top_idx = sorted(int(i) for i in top_idx)
    return " ".join(sentences[i] for i in top_idx)

@lru_cache(maxsize=4)
def get_translation_pipeline(source_code, target_code):
    from transformers import pipeline

    model_map = {
        ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
        ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    }
    model_name = model_map[(source_code, target_code)]
    return pipeline("translation", model=model_name)

def translate_text(text, source_lang, target_lang):
    from_code = SUPPORTED_LANGUAGES[source_lang]
    to_code = SUPPORTED_LANGUAGES[target_lang]
    pipe = get_translation_pipeline(from_code, to_code)
    result = pipe(text, max_length=512)
    return result[0]["translation_text"]
