import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import os

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Configuration
# Path to your fine-tuned model and tokenizer
model_path = "inputs/fine_tuned_model_paper"

input_file = "inputs/crs_all_translated_2000-2023_unique.csv"  # Input file to unique dataset (reduced duplicate texts, added unique ID)

output_file = "outputs/crs_all_translated_2000-2023_unique_binarylabeled.csv"
batch_size = 64  # Batch size for inference
max_length = 200  # Maximum sequence length for tokenization

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model = model.to(device)
model.eval()


# Dataset class for batching
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# Tokenization function
def tokenize_batch(texts):
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens


# Load the dataset
print("Loading dataset...")
df = pd.read_csv(input_file)
total_records = df.shape[0]
print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")
print(df.head())
print(f"Dataset contains {total_records} rows and {df.shape[1]} columns.")

# Ensure no NaN values in the text column
df["text"] = df["text"].fillna("").astype(str)

# Create DataLoader for large dataset
print("Preparing DataLoader...")
dataset = TextDataset(df["text"].tolist())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Perform inference
print("Starting inference...")
all_predictions = []
processed_records = 0  # Counter for processed records

for i, batch in enumerate(dataloader):
    tokens = tokenize_batch(batch)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(preds.cpu().numpy())

    # Update the count of processed records
    processed_records += len(batch)

    # Progress reporting every 100,000 records
    if processed_records % 10_000 <= batch_size:
        percent_complete = (processed_records / total_records) * 100
        print(
            f"{processed_records} records from {total_records} records labeled so far, {percent_complete:.2f}% of data processed."
        )

# Add the new column with predictions
df["binary_label_biodiversity_impact"] = all_predictions  # New column for labels

# Save the full dataset with all original columns and the new label column
print("Saving results...")
df.to_csv(output_file, index=False)
print(
    f"Labeled dataset saved to {output_file}. The dataset contains {df.shape[1]} columns."
)
