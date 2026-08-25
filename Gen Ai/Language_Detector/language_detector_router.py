from transformers import pipeline

print("=" * 60)
print("LANGUAGE DETECTOR AND ROUTER")
print("=" * 60)

# Load Hugging Face model
model_name = "dnivra26/muril-lang-id-v7"

print("\nLoading language detection model...")
detector = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=model_name
)

print("Model loaded successfully!\n")

# Five test messages
messages = [
    "I need help resetting my account password.",
    "मुझे अपने खाते का पासवर्ड रीसेट करना है।",
    "Necesito ayuda con mi pedido.",
    "నా ఆర్డర్ గురించి సహాయం కావాలి.",
    "Enakku en account password reset panna help venum."
]

# Language names
language_names = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "es": "Spanish"
}

# Process messages
for i, message in enumerate(messages, 1):

    result = detector(message)[0]

    label = result["label"].lower()
    confidence = result["score"]

    language = language_names.get(label, label)

    # Routing
    if label == "en":
        team = "Team A"
    else:
        team = "Team B"

    print("-" * 60)
    print(f"MESSAGE {i}")
    print(f"Text       : {message}")
    print(f"Language   : {language} ({label})")
    print(f"Confidence : {confidence:.4f}")
    print(f"Route      : {team}")

print("-" * 60)
print("Detection and routing completed.")