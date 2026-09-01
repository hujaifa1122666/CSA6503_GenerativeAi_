# Q2 – General-Purpose Generative AI Application

## Assignment alignment
A general-purpose Generative AI application supporting text generation, summarization, question answering, translation, and chatbot interaction.

## Implementation
### Chatbot / text generation
Uses `google/flan-t5-base` in a text-to-text generation pipeline.

### Summarization
Uses FLAN-T5 with an explicit summarization instruction.

### Question answering
Uses FLAN-T5 with context + question prompting.

### Translation
Uses `google/mt5-small` with language-specific translation prompts. mT5 is a multilingual T5-family model.

## Run
```powershell
py -m venv C:\venvs\co4at2q2
C:\venvs\co4at2q2\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Expected first-run behavior
Model files are downloaded the first time they are used. This may take several minutes.

## Invalid input handling
The UI handles:
- empty input
- very short input
- empty QA question
- same source and target language

## Limitations
- Small local models may produce weaker answers than large commercial LLMs.
- Translation prompts may not be equally strong for every language pair.
- QA quality depends on the supplied context.
- Summarization may miss details in long documents.
