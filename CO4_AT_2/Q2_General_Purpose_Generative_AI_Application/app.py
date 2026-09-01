import streamlit as st
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="General-Purpose Generative AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# MODEL NAMES
# ---------------------------------------------------------

MODEL_TEXT = "google/flan-t5-base"
MODEL_TRANSLATION = "google/mt5-small"

# ---------------------------------------------------------
# LOAD FLAN-T5
# Used for:
# - Chatbot
# - Text Generation
# - Summarization
# - Question Answering
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_flan():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TEXT
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_TEXT
    )

    return tokenizer, model


# ---------------------------------------------------------
# LOAD mT5
# Used for translation
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_mt5():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TRANSLATION
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_TRANSLATION
    )

    return tokenizer, model


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_text(text):

    if not text or not text.strip():

        return "Input cannot be empty."

    if len(text.strip()) < 3:

        return "Please enter a little more text."

    return None


# ---------------------------------------------------------
# FLAN-T5 GENERATION
# ---------------------------------------------------------

def generate_flan(
    prompt,
    max_tokens=180
):

    tokenizer, model = load_flan()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4
    )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return result


# ---------------------------------------------------------
# mT5 TRANSLATION
# ---------------------------------------------------------

def generate_mt5(
    prompt,
    max_tokens=120
):

    tokenizer, model = load_mt5()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4
    )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return result


# ---------------------------------------------------------
# APPLICATION TITLE
# ---------------------------------------------------------

st.title(
    "🤖 General-Purpose Generative AI Application"
)

st.caption(
    "Chatbot • Summarizer • Question Answering • Translation"
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Select AI Task")

task = st.sidebar.selectbox(
    "Choose a task",
    [
        "Chatbot / Text Generation",
        "Summarization",
        "Question Answering",
        "Translation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Text Generation Model:**"
)

st.sidebar.write(
    "FLAN-T5-base"
)

st.sidebar.write(
    "**Translation Model:**"
)

st.sidebar.write(
    "mT5-small"
)


# =========================================================
# 1. CHATBOT / TEXT GENERATION
# =========================================================

if task == "Chatbot / Text Generation":

    st.header(
        "💬 Chatbot / Text Generation"
    )

    user_input = st.text_area(
        "Enter your prompt or question",
        height=180,
        placeholder="Explain generative AI in simple terms."
    )

    if st.button(
        "Generate Response",
        type="primary",
        use_container_width=True
    ):

        error = validate_text(
            user_input
        )

        if error:

            st.warning(error)

        else:

            with st.spinner(
                "Generating response..."
            ):

                answer = generate_flan(
                    "Answer clearly and helpfully: "
                    + user_input,
                    max_tokens=180
                )

            st.subheader(
                "Response"
            )

            st.write(answer)


# =========================================================
# 2. SUMMARIZATION
# =========================================================

elif task == "Summarization":

    st.header(
        "📝 Text Summarization"
    )

    text = st.text_area(
        "Paste a long article or study notes",
        height=300,
        placeholder="Paste your text here..."
    )

    if st.button(
        "Summarize",
        type="primary",
        use_container_width=True
    ):

        error = validate_text(
            text
        )

        if error:

            st.warning(error)

        else:

            with st.spinner(
                "Generating summary..."
            ):

                summary = generate_flan(
                    "Summarize the following text "
                    "in concise and clear points:\n\n"
                    + text,
                    max_tokens=180
                )

            st.subheader(
                "Summary"
            )

            st.write(summary)


# =========================================================
# 3. QUESTION ANSWERING
# =========================================================

elif task == "Question Answering":

    st.header(
        "❓ Question Answering"
    )

    context = st.text_area(
        "Enter Context / Document",
        height=250,
        placeholder=(
            "Paste the information from which "
            "the answer should be generated."
        )
    )

    question = st.text_input(
        "Enter Your Question",
        placeholder="What is the main idea?"
    )

    if st.button(
        "Answer Question",
        type="primary",
        use_container_width=True
    ):

        context_error = validate_text(
            context
        )

        if context_error:

            st.warning(
                "Context: " + context_error
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            prompt = (
                "Answer the question using only "
                "the following context.\n\n"
                "Context:\n"
                + context
                + "\n\nQuestion:\n"
                + question
            )

            with st.spinner(
                "Finding answer..."
            ):

                answer = generate_flan(
                    prompt,
                    max_tokens=120
                )

            st.subheader(
                "Answer"
            )

            st.write(answer)


# =========================================================
# 4. TRANSLATION
# =========================================================

else:

    st.header(
        "🌐 Machine Translation"
    )

    source_language = st.selectbox(
        "Source Language",
        [
            "English",
            "Hindi",
            "Tamil",
            "Telugu"
        ]
    )

    target_language = st.selectbox(
        "Target Language",
        [
            "English",
            "Hindi",
            "Tamil",
            "Telugu"
        ],
        index=2
    )

    translation_text = st.text_area(
        "Enter Text to Translate",
        height=180,
        placeholder="Enter text here..."
    )

    # -----------------------------------------------------
    # Translation Prompts
    # -----------------------------------------------------

    translation_prompts = {

        ("English", "Hindi"):
            "translate English to Hindi: ",

        ("English", "Tamil"):
            "translate English to Tamil: ",

        ("English", "Telugu"):
            "translate English to Telugu: ",

        ("Hindi", "English"):
            "translate Hindi to English: ",

        ("Tamil", "English"):
            "translate Tamil to English: ",

        ("Telugu", "English"):
            "translate Telugu to English: ",

        ("Hindi", "Tamil"):
            "translate Hindi to Tamil: ",

        ("Tamil", "Hindi"):
            "translate Tamil to Hindi: ",

        ("Hindi", "Telugu"):
            "translate Hindi to Telugu: ",

        ("Telugu", "Hindi"):
            "translate Telugu to Hindi: ",

        ("Tamil", "Telugu"):
            "translate Tamil to Telugu: ",

        ("Telugu", "Tamil"):
            "translate Telugu to Tamil: "
    }

    if st.button(
        "Translate",
        type="primary",
        use_container_width=True
    ):

        error = validate_text(
            translation_text
        )

        if error:

            st.warning(error)

        elif source_language == target_language:

            st.warning(
                "Choose different source and target languages."
            )

        else:

            prompt_key = (
                source_language,
                target_language
            )

            prefix = translation_prompts[
                prompt_key
            ]

            prompt = prefix + translation_text

            with st.spinner(
                "Translating..."
            ):

                result = generate_mt5(
                    prompt,
                    max_tokens=120
                )

            st.subheader(
                "Translation"
            )

            st.write(result)