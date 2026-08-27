import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Q1 Multilingual NLP Model Comparison", page_icon="🌍", layout="wide")

st.title("🌍 Q1 – Multilingual NLP Model Comparison")
st.write("Practical comparison of BERT, GPT, T5 and LLaMA model families for English and Indian-language NLP.")

MODEL_INFO = {
    "BERT": {
        "checkpoint": "bert-base-multilingual-cased",
        "architecture": "Encoder-only Transformer",
        "objective": "Masked Language Modeling",
        "best_for": "Classification, NER, retrieval, embeddings",
        "language_note": "mBERT supports multiple languages and is suitable for multilingual understanding."
    },
    "GPT": {
        "checkpoint": "openai-community/gpt2",
        "architecture": "Decoder-only autoregressive Transformer",
        "objective": "Next-token prediction",
        "best_for": "Text generation and conversational generation",
        "language_note": "GPT-2 is primarily English; multilingual use requires a multilingual GPT-family model."
    },
    "T5": {
        "checkpoint": "google/mt5-small",
        "architecture": "Encoder-decoder Transformer",
        "objective": "Span corruption / text-to-text pretraining",
        "best_for": "Translation, summarization, QA, text transformation",
        "language_note": "mT5 was pretrained across many languages including Hindi, Tamil, Telugu and other Indian languages."
    },
    "LLaMA": {
        "checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "architecture": "Decoder-only autoregressive Transformer",
        "objective": "Next-token prediction + instruction tuning",
        "best_for": "Generation and chat applications",
        "language_note": "This lightweight LLaMA-family checkpoint is used for local demonstration; multilingual coverage varies by checkpoint."
    },
}

st.subheader("Model comparison")
df = pd.DataFrame([
    {
        "Model": name,
        "Checkpoint": info["checkpoint"],
        "Architecture": info["architecture"],
        "Pre-training objective": info["objective"],
        "Typical use": info["best_for"],
    }
    for name, info in MODEL_INFO.items()
])
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Multilingual practical demonstration")

@st.cache_resource
def load_mbert():
    return SentenceTransformer("sentence-transformers/bert-base-multilingual-cased")

languages = {
    "English": "Artificial intelligence helps students learn difficult concepts.",
    "Hindi": "कृत्रिम बुद्धिमत्ता छात्रों को कठिन अवधारणाएँ सीखने में मदद करती है।",
    "Tamil": "செயற்கை நுண்ணறிவு மாணவர்கள் கடினமான கருத்துகளை கற்றுக்கொள்ள உதவுகிறது.",
    "Telugu": "కృత్రిమ మేధస్సు విద్యార్థులకు కష్టమైన భావనలను నేర్చుకోవడానికి సహాయపడుతుంది.",
}

language = st.selectbox("Choose an example language", list(languages))
text = languages[language]
st.text_area("Example text", text, height=100)

if st.button("Generate mBERT Embedding", type="primary"):
    with st.spinner("Loading mBERT..."):
        model = load_mbert()
        emb = model.encode([text], normalize_embeddings=True)
    st.success("Embedding generated successfully.")
    st.write("Embedding dimension:", len(emb[0]))
    st.write("First 10 values:")
    st.code(np.round(emb[0][:10], 4).tolist())

st.markdown("---")
st.subheader("Architecture and suitability")

choice = st.selectbox("Select a model to inspect", list(MODEL_INFO))
info = MODEL_INFO[choice]
st.write("**Architecture:**", info["architecture"])
st.write("**Pre-training objective:**", info["objective"])
st.write("**Best suited for:**", info["best_for"])
st.write("**Multilingual note:**", info["language_note"])

st.info(
    "For a production multilingual application supporting Indian languages, "
    "a multilingual encoder such as mBERT is strong for understanding tasks, "
    "while mT5-style encoder-decoder models are convenient for multilingual "
    "text-to-text tasks such as translation and summarization."
)

st.subheader("Strengths and limitations")
strengths = {
    "BERT": "Strong contextual understanding and classification/retrieval tasks.",
    "GPT": "Strong open-ended text generation and fluent completion.",
    "T5": "Unified text-to-text interface for multiple NLP tasks.",
    "LLaMA": "Efficient open-weight family for local generation and customization.",
}
limitations = {
    "BERT": "Not designed for unrestricted text generation.",
    "GPT": "Small GPT-2 checkpoint is weak for Indian-language generation.",
    "T5": "Sequence-to-sequence inference can be slower than encoder-only models.",
    "LLaMA": "Multilingual quality depends strongly on the selected checkpoint."
}
st.write("**Strength:**", strengths[choice])
st.write("**Limitation:**", limitations[choice])
