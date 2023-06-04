from transformers import pipeline

model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = model("I want to book a flight from New York to London")

intent = result[0]["label"]
confidence = result[0]["score"]
print(f"Intent: {intent}, Confidence: {confidence}")
