import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# SwearWords preprocessing
file = open('dataset/swearwords.txt', 'r')
swearwordsArray = file.read().split('\n')

# NormalWords preprocessing
file = open('dataset/normal_words.txt', 'r')
normalwordsArray = file.read().split('\n')

# Join array of words
dataset = swearwordsArray + normalwordsArray

# Create label array
labels = []
for i in range(len(swearwordsArray)):
  labels.append(1)

for i in range(len(normalwordsArray)):
  labels.append(0)

# Transform dataset to numpy array
dataset = np.array(dataset)
labels = np.array(labels)

# Generate new order of indexes
indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Apply new order for our dataset and labels
shuffled_dataset = dataset[indices]
shuffled_labels = labels[indices]

# Declare tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
model = AutoModelForSequenceClassification.from_pretrained("sberbank-ai/ruBert-base", num_labels=2)

# Tokenization of data
train_encodings = tokenizer(shuffled_dataset.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create a class for data loading
class MatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MatDataset(train_encodings, shuffled_labels.tolist())

# Training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained('./model_directory')
tokenizer.save_pretrained('./model_directory')