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
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings, dtype="float32")


def validate(query):
    if not query or not query.strip():
        return "Please enter a question."

    if len(query.strip()) < 3:
        return "Please enter a more detailed question."

    return None


def answer_query(query, records, embeddings, model, threshold=0.42):
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

    if best_score < threshold:
        return {
            "answer": (
                "I’m not confident that I have the information needed "
                "for this question. Please contact the department office "
                "or provide more details."
            ),
            "score": best_score,
            "category": "Fallback",
            "matched": "No confident match"
        }

    item = records[best_index]

    return {
        "answer": item["answer"],
        "score": best_score,
        "category": item["category"],
        "matched": item["question"]
    }


st.title("🎓 Student Academic Support Chatbot")

st.caption(
    "AI support for subjects, assignments, examinations, "
    "attendance and academic schedules."
)

records = load_knowledge()
model = load_model()
embeddings = build_index(records)


with st.sidebar:
    st.header("Example Questions")

    examples = [
        "What is the minimum attendance requirement?",
        "How can I submit my assignment?",
        "When are internal examinations?",
        "Where can I get my hall ticket?",
        "Where can I find the academic calendar?"
    ]

    for question in examples:
        st.write("•", question)

    st.divider()

    st.write(f"Knowledge base: **{len(records)} FAQs**")
    st.write("Model: `all-MiniLM-L6-v2`")


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
        result = answer_query(
            query,
            records,
            embeddings,
            model
        )

        st.subheader("Response")

        st.write(result["answer"])

        with st.expander("Retrieval Details"):
            st.write(
                f"Category: **{result['category']}**"
            )

            st.write(
                f"Similarity Score: **{result['score']:.3f}**"
            )

            st.write(
                f"Matched FAQ: {result['matched']}"
            )