import chardet
import pandas as pd
import torch
import os
import logging
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging
from sklearn.utils.class_weight import compute_class_weight


# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Detect encoding
data_path = ("inputs/BERT_random_2000_vfinal.csv")
with open(data_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")

# Load dataset
try:
    df = pd.read_csv(data_path, sep= ";", encoding=encoding, encoding_errors='replace')
    print("File loaded successfully!")
    print(df.head())
except UnicodeDecodeError as e:
    print(f"Error reading the file: {e}")
    raise


# Preprocessing
df = df.rename(columns={"Biodiversity binary label": "label"})
df['text'] = df['text'].fillna('').astype(str)

# Seed setup
def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_state = 2022
set_seed(random_state)

# Data split
train_text, temp_text, train_labels, temp_labels = train_test_split(
    df['text'], df['label'], test_size=0.30, random_state=random_state, stratify=df['label']
)
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, test_size=0.50, random_state=random_state, stratify=temp_labels
)

print(f"Training set: {len(train_text)} samples")
print(f"Validation set: {len(val_text)} samples")
print(f"Test set: {len(test_text)} samples")

# Tokenization
def tokenize_data(texts, labels):
    tokens = tokenizer(texts, max_length=n_words, padding='max_length', truncation=True, return_tensors="pt")
    return TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(labels))

model_name = "./NoYo25 BiodivBERT"
n_words = 200
batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

train_dataset = tokenize_data(train_text.tolist(), train_labels.tolist())
val_dataset = tokenize_data(val_text.tolist(), val_labels.tolist())
test_dataset = tokenize_data(test_text.tolist(), test_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Weighted loss (preferred for recall emphasis)
# CHOOSE WEIGHT MODE: 'manual', 'equal', or 'auto',final model was trained with manual weights, see methods in paper
weight_mode = 'manual'  # or 'equal', or 'auto'

# Manual weight values (used only if weight_mode == 'manual')
manual_weights = [1.0, 2.0]



if weight_mode == 'equal':
    weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(device)
elif weight_mode == 'manual':
    weights = torch.tensor(manual_weights, dtype=torch.float).to(device)
elif weight_mode == 'auto':
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print(f"Auto-computed class weights: {class_weights}")
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
else:
    raise ValueError("Invalid weight_mode. Use 'equal', 'manual', or 'auto'.")

loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

# Optional: Uncomment below to try threshold tuning after switching back to unweighted loss
# loss_fn = torch.nn.CrossEntropyLoss()

best_val_f1 = 0.0
patience = 6
patience_counter = 0
num_epochs = 25

current_date = datetime.now().strftime("%Y_%m_%d-%I_%M")
best_model_path = f"outputs/finetuned_BERT_model_{current_date}.pt"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch_seq, batch_mask, batch_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(batch_seq, attention_mask=batch_mask)
        loss = loss_fn(outputs.logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    all_predictions, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            batch_seq, batch_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(batch_seq, attention_mask=batch_mask)
            loss = loss_fn(outputs.logits, batch_labels)
            val_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]
            predictions = (probs >= 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions, average='weighted')
    val_recall = recall_score(all_labels, all_predictions, average='weighted')
    val_f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with Val F1-Score: {best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered due to no improvement in F1-score.")
            break
        print(f"Best model saved with Val F1-Score: {best_val_f1:.4f}")
# Load best model
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("Best model loaded for evaluation on test set.")

all_test_predictions, all_test_labels = [], []
false_positives, false_negatives = [], []

with torch.no_grad():
    for batch in test_loader:
        batch_seq, batch_mask, batch_labels = [b.to(device) for b in batch]
        outputs = model(batch_seq, attention_mask=batch_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]
        threshold = 0.5
        predictions = (probabilities >= threshold).long()

        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_labels.extend(batch_labels.cpu().numpy())

        batch_texts = tokenizer.batch_decode(batch_seq, skip_special_tokens=True)
        for text, pred, label in zip(batch_texts, predictions.cpu().numpy(), batch_labels.cpu().numpy()):
            if pred != label:
                if pred == 1 and label == 0:
                    false_positives.append((text, pred, label))
                elif pred == 0 and label == 1:
                    false_negatives.append((text, pred, label))

# Test metrics
test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
test_precision = precision_score(all_test_labels, all_test_predictions, average='weighted')
test_recall = recall_score(all_test_labels, all_test_predictions, average='weighted')
test_f1 = f1_score(all_test_labels, all_test_predictions, average='weighted')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# Save errors
fp_fn_df = pd.DataFrame(false_positives + false_negatives, columns=["Text", "Prediction", "Actual"])
fp_fn_df.to_excel(f"outputs/false_positives_negatives_test_{current_date}.xlsx", index=False)
print("False positives and negatives saved.")

# Save model and results
model.save_pretrained(f"outputs/fine_tuned_model_{current_date}")
tokenizer.save_pretrained(f"outputs/fine_tuned_model_{current_date}")
results_test = pd.DataFrame([{
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1
}])
results_test.to_csv(f"outputs/test_set_results_{current_date}", index=False)
print("Results and model saved.")

# Optional threshold tuning code (commented)
# from sklearn.metrics import f1_score
# best_threshold = 0.5
# best_f1 = 0
# val_probs = np.array(all_probs)
# val_labels_np = np.array(all_labels)
# for t in np.arange(0.1, 0.9, 0.01):
#     preds = (val_probs >= t).astype(int)
#     f1 = f1_score(val_labels_np, preds)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = t
# print(f"Best threshold for F1: {best_threshold}")
