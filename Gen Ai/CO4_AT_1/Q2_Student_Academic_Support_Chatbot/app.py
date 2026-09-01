import json
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "academic_knowledge.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Student Academic Support Chatbot",
    page_icon="🎓",
    layout="centered"
)


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_knowledge():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def build_index(records):
    model = load_model()

    embeddings = model.encode(
        [r["question"] for r in records],
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype="float32")


def validate(query):
    if not query or not query.strip():
        return "Please enter a question."

    if len(query.strip()) < 3:
        return "Please enter a more detailed question."

    return None


def get_answer(query, records, embeddings, model):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < 0.42:
        return (
            "I’m not confident that I have the information needed "
            "for this question. Please contact the department office "
            "or provide more details.",
            best_score,
            "Fallback"
        )

    item = records[best_index]

    return (
        item["answer"],
        best_score,
        item["category"]
    )


st.title("🎓 Student Academic Support Chatbot")

st.write(
    "Ask questions about subjects, assignments, examinations, "
    "attendance, and academic schedules."
)

records = load_knowledge()
model = load_model()
embeddings = build_index(records)

with st.sidebar:

    st.header("Example Questions")

    st.write(
        "• What is the minimum attendance requirement?"
    )

    st.write(
        "• How can I submit my assignment?"
    )

    st.write(
        "• When are internal examinations?"
    )

    st.write(
        "• Where can I get my hall ticket?"
    )

    st.write(
        "• Where can I find the academic calendar?"
    )

query = st.text_area(
    "Ask your academic question",
    placeholder="Example: What happens if my attendance is low?",
    height=130
)

if st.button(
    "Get Answer",
    type="primary",
    use_container_width=True
):

    error = validate(query)

    if error:
        st.warning(error)

    else:

        answer, score, category = get_answer(
            query,
            records,
            embeddings,
            model
        )

        st.subheader("Response")

        st.write(answer)

        with st.expander("Retrieval Details"):

            st.write(
                f"Category: **{category}**"
            )

            st.write(
                f"Similarity Score: **{score:.3f}**"
            )