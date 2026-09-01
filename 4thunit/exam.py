from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

text = """
Artificial intelligence is one of the fastest-growing technologies in
the world. It is used in healthcare, education, banking, transportation,
software development, and many other industries. AI systems can analyze
large amounts of data, recognize patterns, answer questions, generate
content, and help people make better decisions. Generative AI has become
especially popular because it can create text, images, code, and other
types of content. Students and developers are increasingly using AI tools
to learn new skills and build innovative applications.
"""

prompt = f"""
Summarize the following text in 3 simple sentences.
Keep only the most important information.

TEXT:
{text}

SUMMARY:
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.3
)

print("----- ORIGINAL TEXT -----")
print(text)

print("\n----- SUMMARY -----")
print(response.choices[0].message.content)