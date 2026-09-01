# Install: pip install transformers torch
from transformers import pipeline

question_answerer = pipeline("question-answering")

context = """
Python is a popular programming language.
It is widely used for artificial intelligence,
data science, web development, and automation.
"""

question = "What is Python widely used for?"

answer = question_answerer(
    question=question,
    context=context
)

print("Question:", question)
print("Answer:", answer["answer"])
print("Confidence:", round(answer["score"], 4))
