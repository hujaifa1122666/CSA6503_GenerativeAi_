from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

question = input("Enter your question: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful question-answering assistant. Give simple and clear answers."
        },
        {
            "role": "user",
            "content": question
        }
    ],
    temperature=0.3
)

print("\nAnswer:")
print(response.choices[0].message.content)