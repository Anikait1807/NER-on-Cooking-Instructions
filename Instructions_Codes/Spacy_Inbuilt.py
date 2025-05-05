import spacy
import pandas as pd
import random
from spacy.training.example import Example
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
file_path = "/content/Modified_Oversampled_Phrases_Sort.xlsx"
data = pd.read_excel(file_path).iloc[1:]  # Remove first row if necessary

# Define label categories from dataset
LABELS = ["QTY", "UNIT", "ING", "FORM", "STATE", "FRESH", "SIZE", "PREPROC"]

# *🔹 Function to Remove Overlapping Entities*
def remove_overlapping_entities(entities):
    """ Keeps only the longest entity when there's an overlap """
    sorted_entities = sorted(entities, key=lambda x: (x[0], -x[1]))  # Sort by start, then by longest
    filtered_entities = []
    prev_start, prev_end = -1, -1

    for start, end, label in sorted_entities:
        if start >= prev_end:  # No overlap, add it
            filtered_entities.append((start, end, label))
            prev_start, prev_end = start, end  # Update previous

    return filtered_entities

# *🔹 Convert Data into spaCy Training Format*
def convert_data_spacy(data):
    processed_data = []
    for _, row in data.iterrows():
        text = row[0]  # Assuming first column contains the sentence
        entities = []

        # Extract entity information
        for i, label in enumerate(LABELS, start=1):
            entity_value = row[i]  # Assuming columns 1 to N contain entity values
            if pd.notna(entity_value):
                start = text.find(str(entity_value))
                if start != -1:
                    end = start + len(str(entity_value))
                    entities.append((start, end, label))

        # *Remove overlapping entities*
        entities = remove_overlapping_entities(entities)

        processed_data.append((text, {"entities": entities}))

    return processed_data

# *🔹 Train-Test Split (80-20)*
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# *🔹 Convert Train & Test Data*
train_data_processed = convert_data_spacy(train_data)
test_data_processed = convert_data_spacy(test_data)

# *🔹 Load blank spaCy model*
nlp = spacy.blank("en")

# *🔹 Add NER pipeline*
ner = nlp.add_pipe("ner")

# *🔹 Add entity labels to NER model*
for label in LABELS:
    ner.add_label(label)

# *🔹 Training Configuration*
optimizer = nlp.begin_training()
for epoch in range(10):  # Train for 10 epochs
    random.shuffle(train_data_processed)
    losses = {}
    for text, annotations in train_data_processed:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Epoch {epoch + 1}, Loss: {losses}")

# *🔹 Evaluation Function*
def evaluate_model(nlp, dataset):
    true_labels, pred_labels = [], []

    for text, annotations in dataset:
        doc = nlp(text)
        pred_ents = {ent.text: ent.label_ for ent in doc.ents}

        for start, end, label in annotations["entities"]:
            true_labels.append(label)
            pred_labels.append(pred_ents.get(text[start:end], "O"))  # Default to "O" if not found

    print("Evaluation Metrics:")
    print(classification_report(true_labels, pred_labels))

# *🔹 Evaluate on Train & Test Sets*
print("\n📌 Evaluation on Train Dataset:")
evaluate_model(nlp, train_data_processed)

print("\n📌 Evaluation on Test Dataset:")
evaluate_model(nlp, test_data_processed)
