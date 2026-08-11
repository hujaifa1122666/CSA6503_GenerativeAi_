from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": """
            Artificial intelligence is changing the way people learn, work,
            and solve everyday problems. AI systems can analyze large amounts
            of information, answer questions, generate content, and assist
            developers in building applications.
            """
        }
    ],
    temperature=0.8
)

print(resp.choices[0].message.content)