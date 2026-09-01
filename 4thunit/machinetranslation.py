from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

text = input("Enter the text to translate: ")
language = input("Enter the target language: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a machine translation assistant. Translate the user's text accurately into the requested language. Give only the translated text."
        },
        {
            "role": "user",
            "content": f"Translate this text into {language}:\n\n{text}"
        }
    ],
    temperature=0.2
)

print("\nTranslated Text:")
print(response.choices[0].message.content)