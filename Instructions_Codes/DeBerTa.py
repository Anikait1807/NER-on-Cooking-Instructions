import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizerFast, DebertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

# ---------------------------
# 1. Load the Dataset
# ---------------------------
# Update the file path as needed (for example, if you upload your file to Colab, it might be in /content/)
file_path = '/content/Modified_Oversampled_Phrases_Sort.xlsx'
data = pd.read_excel(file_path)
data_cleaned = data.iloc[1:]  # removing the first row if needed

# ---------------------------
# 2. Split the Dataset
# ---------------------------
train_data, test_data = train_test_split(data_cleaned, test_size=0.20, random_state=42)

# ---------------------------
# 3. Setup Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 4. Initialize the Tokenizer
# ---------------------------
tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')

# ---------------------------
# 5. Define the Label Dictionary
# ---------------------------
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

# ---------------------------
# 6. Create a Custom Dataset Class
# ---------------------------
class IngredientDataset(Dataset):
    def _init_(self, dataframe, tokenizer, label_dict, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.max_len = max_len

    def _getitem_(self, index):
        # Assume first column contains the sentence/text
        sentence = self.data.iloc[index, 0]
        # Create a list of labels for the tokens.
        # (Assumes subsequent columns contain label information; adjust as needed)
        word_labels = self.create_labels(self.data.iloc[index], self.label_dict)

        # Tokenize the sentence with offset mappings
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # Create a labels array and assign label -100 to special tokens (no loss computed)
        labels = [self.label_dict['O']] * self.max_len
        arr_offset = encoding['offset_mapping']

        for idx, offset in enumerate(arr_offset):
            if offset[0] == offset[1]:
                labels[idx] = -100
            else:
                # Use the corresponding label if available; otherwise default to 'O'
                labels[idx] = word_labels[idx] if idx < len(word_labels) else self.label_dict['O']

        encoding['labels'] = labels
        # Convert to torch tensors and filter only necessary keys
        encoding = {k: torch.as_tensor(v) for k, v in encoding.items() if k in ['input_ids', 'attention_mask', 'labels']}
        return encoding

    def _len_(self):
        return self.len

    def create_labels(self, data_row, label_dict):
        labels = []
        # For each label type, extract the entity from the row.
        # This sample assumes that the first column is text and the next columns are entities.
        # Adjust the indexing if your dataset structure is different.
        for col_index, (label_base, _) in enumerate(label_dict.items()):
            # In this example we skip half of the entries for a simplistic mapping.
            if col_index % 2 == 0:
                continue
            # Adjust the column index (+1 because first column is the text)
            entity = data_row.iloc[col_index // 2 + 1]
            if pd.notna(entity):
                labels.extend([label_dict[label_base], label_dict['I' + label_base[1:]]])
            else:
                labels.append(label_dict['O'])
        # Extend or trim labels to cover max_len tokens
        return labels * (self.max_len // len(labels) + 1)

# ---------------------------
# 7. Create DataLoaders
# ---------------------------
max_len = 128
batch_size = 16

train_dataset = IngredientDataset(train_data, tokenizer, label_dict, max_len)
test_dataset = IngredientDataset(test_data, tokenizer, label_dict, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# 8. Initialize the DeBERTa Model for Token Classification
# ---------------------------
model = DebertaForTokenClassification.from_pretrained('microsoft/deberta-base', num_labels=len(label_dict))
model.to(device)

# ---------------------------
# 9. Set Up the Optimizer
# ---------------------------
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# ---------------------------
# 10. Define Training and Evaluation Functions
# ---------------------------
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
            # Convert indices back to label names, ignoring positions with label -100
            for lbls, prds in zip(batch_labels, preds):
                for l, p in zip(lbls, prds):
                    if l.item() != -100:
                        true_labels.append(label_dict_inv[l.item()])
                        predictions.append(label_dict_inv[p.item()])
    return classification_report(true_labels, predictions, digits = 4)

# ---------------------------
# 11. Train the Model
# ---------------------------
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_model(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}")

# ---------------------------
# 12. Evaluate the Model
# ---------------------------
print("Evaluating on Training Set...")
train_report = evaluate_model(model, train_loader, device)
print("Training Performance:")
print(train_report)

print("Evaluating on Test Set...")
test_report = evaluate_model(model, test_loader, device)
print("Testing Performance:")
print(test_report)
