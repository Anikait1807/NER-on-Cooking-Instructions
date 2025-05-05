import spacy
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load dataset
file_path = "/content/Modified_Oversampled_Phrases_Sort.xlsx"
data = pd.read_excel(file_path).iloc[1:]  # Remove first row if it's a header

# Initialize tokenizer and model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define all labels in BIO format
LABELS = ["QTY", "UNIT", "ING", "FORM", "STATE", "FRESH", "SIZE", "PREPROC"]
label_dict = {"O": 0}  # Initialize with 'Outside' class

# Add BIO labels to dictionary
for label in LABELS:
    label_dict[f"B-{label}"] = len(label_dict)
    label_dict[f"I-{label}"] = len(label_dict)

label_dict_inv = {v: k for k, v in label_dict.items()}  # Reverse mapping

# Convert dataset into tokenized format with BIO labels
def convert_data_spacy_transformers(data):
    processed_data = []
    for _, row in data.iterrows():
        text = row[0]  # Assuming first column has the sentence
        tokens = tokenizer(text, return_offsets_mapping=True, truncation=True)
        labels = ["O"] * len(tokens["input_ids"])

        # Assign BIO labels
        for i, label in enumerate(LABELS, start=1):
            entity = row[i]  # Extract entity from dataset
            if pd.notna(entity):
                start = text.find(str(entity))
                if start != -1:
                    end = start + len(str(entity))
                    for j, (s, e) in enumerate(tokens["offset_mapping"]):
                        if s == start:
                            labels[j] = f"B-{label}"
                        elif s > start and e <= end:
                            labels[j] = f"I-{label}"

        encoded_labels = [label_dict[l] for l in labels]
        processed_data.append({"input_ids": tokens["input_ids"], "labels": encoded_labels})

    return processed_data

# *🔹 Train-Test Split (80-20)*
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert train & test data
train_data_processed = convert_data_spacy_transformers(train_data)
test_data_processed = convert_data_spacy_transformers(test_data)

# Create PyTorch Dataset
class NERDataset(torch.utils.data.Dataset):
    def _init_(self, data):
        self.data = data

    def _getitem_(self, index):
        item = self.data[index]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "labels": torch.tensor(item["labels"]),
        }

    def _len_(self):
        return len(self.data)

# Padding function
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored by loss function

    return {"input_ids": input_ids, "labels": labels}

# Create train & test datasets
train_dataset = NERDataset(train_data_processed)
test_dataset = NERDataset(test_data_processed)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Load RoBERTa for token classification
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_dict))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# *🔹 Training Loop*
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# *🔹 Evaluation Function*
def evaluate_model(model, data_loader, label_dict_inv):
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)

            # Convert model output to predictions
            preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
            true_vals = batch["labels"].cpu().numpy()

            # Convert label indices to label names
            for i in range(true_vals.shape[0]):  # Loop over batch size
                for j in range(len(true_vals[i])):  # Loop over tokens
                    if true_vals[i][j] != -100:  # Ignore padding
                        true_labels.append(label_dict_inv.get(true_vals[i][j], "O"))
                        pred_labels.append(label_dict_inv.get(preds[i][j], "O"))

    print("Evaluation Metrics:")
    print(classification_report(true_labels, pred_labels, digits = 4))

# *🔹 Evaluate on Train and Test Sets*
print("\n📌 Evaluation on Train Dataset:")
evaluate_model(model, train_loader, label_dict_inv)

print("\n📌 Evaluation on Test Dataset:")
evaluate_model(model, test_loader, label_dict_inv)
