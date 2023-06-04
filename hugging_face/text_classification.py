from transformers import pipeline

model = pipeline("text-classification",
                 model="distilbert-base-uncased",
                 tokenizer="distilbert-base-uncased",
                 return_all_scores=True)

text = "I want to cancel a flight from New York to London"
labels = ["book_flight", "cancel_flight", "update_book", "remove_book"]

result = model(text, labels)
print(result)
scores = result[0]["scores"]

for i, label in enumerate(labels):
    print(f"Intent: {label}, Confidence: {scores[i]}")
