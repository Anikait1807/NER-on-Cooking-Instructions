import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

# Load the dataset
file_path = '/content/Modified_Oversampled_Phrases_Sort.xlsx'
data = pd.read_excel(file_path)
data_cleaned = data.iloc[1:]  # removing the first row

# Splitting the dataset into train and test sets
train_data, test_data = train_test_split(data_cleaned, test_size=0.20, random_state=42)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Define the label dictionary
label_dict = {
    'O': 0,  # Outside any named entity
    'B-QTY': 1, 'I-QTY': 2,
    'B-UNIT': 3, 'I-UNIT': 4,
    'B-ING': 5, 'I-ING': 6,  # ING - Ingredient Name
    'B-FORM': 7, 'I-FORM': 8,
    'B-STATE': 9, 'I-STATE': 10,
    'B-FRESH': 11, 'I-FRESH': 12,
    'B-SIZE': 13, 'I-SIZE': 14,
    'B-PREPROC': 15, 'I-PREPROC': 16
}
label_dict_inv = {v: k for k, v in label_dict.items()}

class IngredientDataset(Dataset):
    def _init_(self, dataframe, tokenizer, label_dict, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.max_len = max_len

    def _getitem_(self, index):
        sentence = self.data.iloc[index, 0]
        word_labels = self.create_labels(self.data.iloc[index], label_dict)

        encoding = self.tokenizer(sentence, return_offsets_mapping=True,
                                  padding='max_length', truncation=True,
                                  max_length=self.max_len)
        labels = [self.label_dict['O']] * self.max_len
        arr_offset = encoding['offset_mapping']

        for idx, offset in enumerate(arr_offset):
            if offset[0] == offset[1]:
                labels[idx] = -100
            else:
                labels[idx] = word_labels[idx] if idx < len(word_labels) else self.label_dict['O']

        encoding['labels'] = labels
        encoding = {k: torch.as_tensor(v) for k, v in encoding.items() if k in ['input_ids', 'attention_mask', 'labels']}
        return encoding

    def _len_(self):
        return self.len

    def create_labels(self, data_row, label_dict):
        labels = []
        # Handling labels for each type of information in the row
        for col_index, (label_base, _) in enumerate(label_dict.items()):
            if col_index % 2 == 0:  # Skip 'I-' type labels
                continue
            entity = data_row.iloc[col_index // 2 + 1]  # +1 to adjust for zero-based index and shift by one for 'Column1'
            if pd.notna(entity):
                labels.extend([label_dict[label_base], label_dict['I' + label_base[1:]]])
            else:
                labels.append(label_dict['O'])
        return labels * (max_len // len(labels) + 1)  # Extend labels to cover max_len

max_len = 128
train_dataset = IngredientDataset(train_data, tokenizer, label_dict, max_len)
test_dataset = IngredientDataset(test_data, tokenizer, label_dict, max_len)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(label_dict))
model.to(device)

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            batch_labels = batch['labels']
            # Remove ignored index (special tokens)
            true_labels.extend([label_dict_inv[l.item()] for lbls, prds in zip(batch_labels, preds) for l, p in zip(lbls, prds) if l != -100])
            predictions.extend([label_dict_inv[p.item()] for lbls, prds in zip(batch_labels, preds) for l, p in zip(lbls, prds) if l != -100])
    return classification_report(true_labels, predictions)

# Train the model
epochs = 10  # Updated to 20 epochs
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_model(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")

# Evaluate the model
train_report = evaluate_model(model, train_loader, device)
test_report = evaluate_model(model, test_loader, device)

print("Training Performance:")
print(train_report)
print("Testing Performance:")
print(test_report)