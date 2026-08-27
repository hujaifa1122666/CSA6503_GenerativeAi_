import streamlit as st
from core import (
    chatbot_answer,
    summarize_text,
    translate_text,
    validate_text,
    SUPPORTED_LANGUAGES,
)

st.set_page_config(
    page_title="Intelligent AI Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Intelligent AI Assistant")
st.caption("Chatbot • Summarizer • Translator")

st.write(
    "Select a task, enter your content, and process it through a lightweight local AI pipeline."
)

task = st.sidebar.selectbox(
    "Select Task",
    ["Chatbot", "Summarizer", "Translator"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Techniques")
st.sidebar.write("• Chatbot: semantic retrieval")
st.sidebar.write("• Summarizer: extractive TextRank")
st.sidebar.write("• Translator: local MarianMT model")

if task == "Chatbot":
    st.header("💬 Academic AI Chatbot")
    query = st.text_area(
        "Ask a question",
        placeholder="Example: How can I submit my assignment?",
        height=150,
    )
    if st.button("Ask", type="primary", use_container_width=True):
        err = validate_text(query)
        if err:
            st.warning(err)
        else:
            result = chatbot_answer(query)
            st.subheader("Answer")
            st.write(result["answer"])
            with st.expander("Retrieval details"):
                st.write(f"Category: **{result['category']}**")
                st.write(f"Similarity: **{result['score']:.3f}**")

elif task == "Summarizer":
    st.header("📝 Text Summarizer")
    text = st.text_area(
        "Paste a long document or notes",
        height=330,
        placeholder="Paste several paragraphs here...",
    )
    ratio = st.slider("Summary length", min_value=0.2, max_value=0.6, value=0.35, step=0.05)
    if st.button("Summarize", type="primary", use_container_width=True):
        err = validate_text(text)
        if err:
            st.warning(err)
        else:
            summary = summarize_text(text, ratio=ratio)
            st.subheader("Summary")
            st.write(summary)
            st.caption("Method: TextRank-style sentence ranking using sentence embeddings.")

else:
    st.header("🌐 Machine Translator")
    source_lang = st.selectbox("Source language", list(SUPPORTED_LANGUAGES.keys()))
    target_lang = st.selectbox("Target language", list(SUPPORTED_LANGUAGES.keys()), index=1)
    text = st.text_area(
        "Enter text",
        height=220,
        placeholder="Type or paste text to translate...",
    )
    st.info("Local translation models are downloaded automatically on first use.")
    if st.button("Translate", type="primary", use_container_width=True):
        err = validate_text(text)
        if err:
            st.warning(err)
        elif source_lang == target_lang:
            st.warning("Choose two different languages.")
        else:
            with st.spinner("Loading translation model and translating..."):
                try:
                    result = translate_text(text, source_lang, target_lang)
                    st.subheader("Translation")
                    st.write(result)
                except Exception as exc:
                    st.error(
                        "Translation failed. Check your internet connection for first-time "
                        "model download, then retry."
                    )
                    with st.expander("Technical details"):
                        st.code(str(exc))
