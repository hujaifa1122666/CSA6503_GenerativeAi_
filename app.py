import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "help_articles.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.35

@st.cache_resource
def load_kb():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [a["title"] + ". " + a["content"] for a in articles]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return articles, model, index


def retrieve(query, articles, model, index, top_k=TOP_K):
    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, ids = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({
                "article": articles[idx],
                "score": float(score)
            })
    return results


def generate_answer(query, results):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        # Grounded fallback: useful for testing without an API key.
        steps = []
        for item in results:
            steps.append(
                f"{item['article']['title']}: {item['article']['content']}"
            )
        return (
            "Grounded retrieval result (Gemini API key not configured):\n\n"
            + "\n\n".join(steps)
        )

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        [
            f"[Source {i+1}] {r['article']['title']}\n{r['article']['content']}"
            for i, r in enumerate(results)
        ]
    )

    prompt = f"""
You are a customer-support knowledge-base assistant.

Answer the user's question ONLY from the supplied help-center articles.
Do not invent policies, steps, limits, URLs, or product behavior.
If the question requires information from more than one article, combine
the relevant steps in a logical order and cite both sources.
If the retrieved articles do not contain enough information, say:
"I cannot answer this from the available help-center articles."

User question:
{query}

Retrieved help-center articles:
{context}

Return:
1. A concise, practical answer.
2. Numbered steps when the user needs troubleshooting instructions.
3. Source citations such as [Source 1] and [Source 2].
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )
    return response.text.strip()


st.set_page_config(page_title="Customer Support Knowledge-Base Bot", page_icon="💬")
st.title("💬 Customer Support Knowledge-Base Bot")
st.caption("RAG Document QA over 16 help-center articles")

articles, model, index = load_kb()

st.sidebar.header("Knowledge Base")
st.sidebar.write(f"Articles indexed: {len(articles)}")
st.sidebar.write(f"Embedding model: {EMBEDDING_MODEL}")
st.sidebar.write(f"Top-K retrieval: {TOP_K}")

query = st.text_area(
    "Ask a support question",
    placeholder="Example: I forgot my password and after resetting it I need to change my email address."
)

if st.button("Ask Support Bot") and query.strip():
    results = retrieve(query, articles, model, index)

    if not results or results[0]["score"] < SIMILARITY_THRESHOLD:
        st.warning("I cannot answer this from the available help-center articles.")
    else:
        answer = generate_answer(query, results)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Documentation")
        for i, item in enumerate(results, start=1):
            article = item["article"]
            with st.expander(
                f"Source {i}: {article['title']} | similarity={item['score']:.3f}"
            ):
                st.write(f"Article ID: {article['id']}")
                st.write(article["content"])

st.divider()
st.subheader("Demonstration Queries")
st.markdown(
    """
- **Troubleshooting:** I cannot log in because my account is locked. What should I do?
- **Billing:** My payment failed. What should I check?
- **Combined two-article query:** I forgot my password and after resetting it I need to change my email address. What should I do?
"""
)
