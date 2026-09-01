import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_LANGUAGES = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
}

FAQS = [
    (
        "Attendance",
        "What is the minimum attendance requirement?",
        "Students are expected to maintain the attendance percentage prescribed by the institution."
    ),
    (
        "Assignments",
        "How can I submit an assignment?",
        "Submit assignments through the platform specified by your faculty and follow the deadline and file-format instructions."
    ),
    (
        "Examinations",
        "How do I register for an examination?",
        "Use the official examination-registration portal during the notified registration window."
    ),
    (
        "Examinations",
        "Where can I get my hall ticket?",
        "Check the official examination or student portal for the hall-ticket download link."
    ),
    (
        "Academic Schedule",
        "Where can I find the academic calendar?",
        "Check the official academic portal or institution website for the latest academic calendar."
    ),
    (
        "Subjects",
        "Where can I get my course syllabus?",
        "Download the current syllabus from the official department or student portal."
    ),
    (
        "Assignments",
        "What happens if I miss an assignment deadline?",
        "Contact the concerned faculty member promptly. Late-submission rules depend on the course and faculty policy."
    ),
    (
        "Academic Support",
        "Who should I contact about an academic issue?",
        "Start with the subject faculty or class advisor and escalate to the department office when necessary."
    ),
]


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_faq_index():
    model = get_embedding_model()
    questions = [item[1] for item in FAQS]

    embeddings = model.encode(
        questions,
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype="float32")


def validate_text(text):
    if not text or not text.strip():
        return "Input cannot be empty."

    if len(text.strip()) < 3:
        return "Please enter at least a few meaningful words."

    return None


def chatbot_answer(query, threshold=0.40):
    model = get_embedding_model()
    index = get_faq_index()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        index
    )[0]

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < threshold:
        return {
            "answer": "I do not have enough information to answer confidently. Please provide more details or contact the concerned department.",
            "score": best_score,
            "category": "Fallback",
        }

    category, question, answer = FAQS[best_index]

    return {
        "answer": answer,
        "score": best_score,
        "category": category,
    }


def split_sentences(text):
    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip()
    )

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def summarize_text(text, ratio=0.35):
    sentences = split_sentences(text)

    if len(sentences) <= 2:
        return text.strip()

    model = get_embedding_model()

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        embeddings,
        embeddings
    )

    scores = np.ones(
        len(sentences),
        dtype="float32"
    )

    damping = 0.85

    for _ in range(25):

        new_scores = np.ones(
            len(sentences),
            dtype="float32"
        ) * (1 - damping)

        for i in range(len(sentences)):

            weights = similarity[i].copy()
            weights[i] = 0
            weights = np.maximum(weights, 0)

            total = weights.sum()

            if total == 0:
                continue

            for j in range(len(sentences)):

                if i != j:

                    new_scores[i] += (
                        damping
                        * (weights[j] / total)
                        * scores[j]
                    )

        scores = new_scores

    count = max(
        1,
        min(
            len(sentences),
            int(round(len(sentences) * ratio))
        )
    )

    top_indices = np.argsort(scores)[-count:]

    top_indices = sorted(
        int(index)
        for index in top_indices
    )

    return " ".join(
        sentences[index]
        for index in top_indices
    )


@lru_cache(maxsize=2)
def load_translation_model():

    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


def translate_text(
    text,
    source_lang,
    target_lang
):

    source_code = SUPPORTED_LANGUAGES[source_lang]
    target_code = SUPPORTED_LANGUAGES[target_lang]

    tokenizer, model = load_translation_model()

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    target_token_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        max_length=512
    )

    return tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]