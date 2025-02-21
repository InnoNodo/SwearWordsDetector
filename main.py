import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('./model_directory')
tokenizer = AutoTokenizer.from_pretrained('./model_directory')

test_message = ['хлопок']
test_encodings = tokenizer(test_message, truncation=True, padding=True, max_length=128, return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

if predictions == 1:
  print("Это мат")
else:
  print("Это не мат")