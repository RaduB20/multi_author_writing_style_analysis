# Later we will turn this into a jupyter notebook

# Multi Author Writing Style Analysis
# Authors: Juan Arturo Abaurrea Calafell, Radu-Andrei Bourceanu

# 3 parts: EDA, Training, Evaluation

# Part 1: Exploratory Data Analysis (EDA)

# ----- Code cell 1 -----

import os
import requests
import zipfile

DATA_URL = "https://zenodo.org/records/14891299/files/pan25-multi-author-analysis.zip"
DATA_DIR = "data"
ZIP_FILE_NAME = "pan25-multi-author-analysis.zip"
ZIP_FILE_PATH = os.path.join(DATA_DIR, ZIP_FILE_NAME)
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(ZIP_FILE_PATH):
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    with open(ZIP_FILE_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {ZIP_FILE_PATH}")

if not all(level in os.listdir(DATA_DIR) for level in DIFFICULTY_LEVELS):
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print(f"Extracted to {DATA_DIR}")

# ----- Code cell 2 -----

ALWAYS_PARSE = False  # Set to True to always re-parse the data
DATA_LOADING_PARAMETERS = {
    "difficulty_levels": DIFFICULTY_LEVELS[0],
    "load_rows": 10000  # Set to None to load all rows
}

import pandas as pd
import json

def load_data(data_dir: str, difficulty_levels: list[str] | str, load_rows: int = None) -> pd.DataFrame:
    """
    Loads data from the specified directory and difficulty levels and returns it as a single dataframe.
    Returns a dataframe with columns: sentence1, sentence2, label.
    """
    if not isinstance(difficulty_levels, (list, set, tuple)):
        difficulty_levels = [difficulty_levels]
    
    rows = []
    skipped_count = 0
    
    # Process in the specific order: train, validation, test
    for split in os.listdir(os.path.join(data_dir, difficulty_levels[0])):
        for level in difficulty_levels:
            print(f"Processing split: {split}, level: {level}")
            split_dir = os.path.join(data_dir, level, split)
            
            if not os.path.exists(split_dir):
                continue
            
            problem_files = sorted([f for f in os.listdir(split_dir) if f.startswith('problem-') and f.endswith('.txt')])
            
            for idx, problem_file in enumerate(problem_files):
                if load_rows is not None and idx >= load_rows:
                    break
                
                problem_num = problem_file.replace('problem-', '').replace('.txt', '')
                truth_file = f'truth-problem-{problem_num}.json'
                
                problem_path = os.path.join(split_dir, problem_file)
                truth_path = os.path.join(split_dir, truth_file)
                
                if not os.path.exists(truth_path):
                    skipped_count += 1
                    continue
                
                with open(problem_path, 'r', encoding='utf-8') as f:
                    sentences = [line.strip() for line in f.readlines() if line.strip()]
                
                with open(truth_path, 'r', encoding='utf-8') as f:
                    truth_data = json.load(f)
                    changes = truth_data.get('changes', [])
                
                if len(sentences) != len(changes) + 1:
                    skipped_count += 1
                    continue
                
                for i in range(len(sentences) - 1):
                    rows.append({
                        'sentence1': sentences[i],
                        'sentence2': sentences[i + 1],
                        'label': changes[i] != 0  # Storing as boolean (label: True if changed, False otherwise)
                    })
    
    df = pd.DataFrame(rows)

    df['sentence1'] = df['sentence1'].astype('string')
    df['sentence2'] = df['sentence2'].astype('string')
    df['label'] = df['label'].astype('boolean')
    
    print(f"Total documents skipped due to mismatches: {skipped_count}")
    
    return df

csv_path = os.path.join(DATA_DIR, 'loaded_data.csv')
if not os.path.exists(csv_path) and not ALWAYS_PARSE:
    df = load_data(DATA_DIR, **DATA_LOADING_PARAMETERS)
    df.to_csv(csv_path, index=False)
    print(f"Saved loaded data to {csv_path}\n")
else:
    df = pd.read_csv(csv_path, dtype={'sentence1': 'string', 'sentence2': 'string', 'label': 'boolean'})
    print(f"Loaded data from {csv_path}\n")

print(f"{df.head()}\n")
print(f"{df.info()}\n")
print(f"Label value counts:\n{df['label'].value_counts()}\n")

shortest_len = min(df['sentence1'].str.len().min(), df['sentence2'].str.len().min())
longest_sent1 = df['sentence1'].str.len().max()
longest_sent2 = df['sentence2'].str.len().max()
longest_len = max(longest_sent1, longest_sent2)
longest_combined = longest_sent1 + longest_sent2
print(f"Shortest sentence length: {shortest_len}")
print(f"Longest sentence length: {longest_len}")
print(f"Longest combined sentence pair length: {longest_combined}")

# Most labels are False, indicating that most sentence pairs do not have changes.

# ----- Code cell 3 -----

from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.7 # 70% for training
VALIDATION_TEST_RATIO = 0.5 # of the remaining 30%, split equally between validation and test, i.e., 15% each
RANDOM_SEED = 42

train_df, temp_df = train_test_split(df, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, stratify=df['label'])
validation_df, test_df = train_test_split(temp_df, train_size=VALIDATION_TEST_RATIO, random_state=RANDOM_SEED, stratify=temp_df['label'])

print(f"Training set size: {len(train_df)}. Authorship change rate: {train_df['label'].mean():.2%}")
print(f"Validation set size: {len(validation_df)}. Authorship change rate: {validation_df['label'].mean():.2%}")
print(f"Test set size: {len(test_df)}. Authorship change rate: {test_df['label'].mean():.2%}")

# ----- Code cell 4 -----

# Part 2: Training the model

# We will try both with a normal topology and with a siamese topology

# https://huggingface.co/spaces/mteb/leaderboard
# BERT: DeBERTaV3, ModernBERT, NeoBERT, RoBERTa, BERT
# https://huggingface.co/infgrad/Jasper-Token-Compression-600M
# https://huggingface.co/Qwen/Qwen3-Embedding-8B
# https://huggingface.co/Qwen/Qwen3-Embedding-4B

# ----- Code cell 5 -----

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ----- Code cell 6 -----

# ----- Code cell 7 -----

# Model configuration
MODEL_NAME = 'prajjwal1/bert-mini'  # A really small model to test the code
# Then change it to microsoft/deberta-v3-small
MAX_LENGTH = 256  # Maximum sequence length
BATCH_SIZE = 16  # Adjust based on your GPU memory (increase for more memory)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
OUTPUT_DIR = './results/deberta-authorship'

# Load tokenizer and model
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,  # Binary classification
    problem_type="single_label_classification"
)

# Move model to GPU
model = model.to(device)
print(f"Model loaded with {model.num_parameters():,} parameters")


# Determining appropriate MAX_LENGTH based on data

# Character length vs token length (approximate: 1 token = 4 characters)
approx_max_tokens = longest_combined / 4
print(f"Approximate max tokens needed: {approx_max_tokens:.0f}")

def get_token_length(text1, text2):
    tokens = tokenizer(text1, text2, truncation=False)
    return len(tokens['input_ids'])

# Sample some pairs to see actual token counts
sample_indices = df.sample(min(1000, len(df)), random_state=RANDOM_SEED).index
token_lengths = []
for idx in sample_indices:
    length = get_token_length(df.loc[idx, 'sentence1'], df.loc[idx, 'sentence2'])
    token_lengths.append(length)

token_lengths = pd.Series(token_lengths)
print(f"\nToken length statistics:")
print(f"  Mean: {token_lengths.mean():.0f}")
print(f"  Median: {token_lengths.median():.0f}")
print(f"  95th percentile: {token_lengths.quantile(0.95):.0f}")
print(f"  99th percentile: {token_lengths.quantile(0.99):.0f}")
print(f"  Max: {token_lengths.max()}")

# Check what percentage would be truncated at different lengths
for length in [256, 384, 512, 768]:
    pct_truncated = (token_lengths > length).mean() * 100
    print(f"  {pct_truncated:.2f}% would be truncated at MAX_LENGTH={length}")


# ----- Code cell 8 -----

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(validation_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

def tokenize_function(examples):
    """
    Tokenizes sentence pairs for the model.
    DeBERTa will use [CLS] sentence1 [SEP] sentence2 [SEP] format.
    """
    return tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None  # Let the trainer handle batching
    )

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("Tokenization complete!")

# ----- Code cell 9 -----

def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, F1, and AUC-ROC.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(labels, probs)
    except:
        auc_roc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

# ----- Code cell 10 -----
if __name__ == '__main__': # when turned into a jupyter notebook, this will be removed and dataloader_num_workers set to 0. Until then, this prevents issues on Windows.
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,  # 10% of training steps for warmup
        lr_scheduler_type='cosine',  # Options: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial'
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,  # Only keep 2 best checkpoints
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        dataloader_num_workers=4,  # Parallel data loading
        report_to='none',  # Disable wandb/tensorboard if not needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    print(f"Total training steps: {len(train_dataset) // BATCH_SIZE * NUM_EPOCHS}")

    # ----- Code cell 11 -----

    # Train the model
    train_result = trainer.train()

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")

    # ----- Code cell 12 -----

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate(val_dataset)

    print("\nValidation Results:")
    print(f"  Accuracy:  {val_results['eval_accuracy']:.4f}")
    print(f"  Precision: {val_results['eval_precision']:.4f}")
    print(f"  Recall:    {val_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {val_results['eval_f1']:.4f}")
    print(f"  AUC-ROC:   {val_results['eval_auc_roc']:.4f}")

    # ----- Code cell 13 -----

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print("\nTest Results:")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"  AUC-ROC:   {test_results['eval_auc_roc']:.4f}")

    # ----- Code cell 14 -----

    # Save the final model
    model.save_pretrained(f'{OUTPUT_DIR}/final_model')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final_model')
    print(f"\nModel saved to {OUTPUT_DIR}/final_model")

    # ----- Code cell 15 -----

    # Part 3: Inference on new data

    def predict_authorship_change(sentence1: str, sentence2: str, model, tokenizer, device):
        """
        Predict whether there's an authorship change between two sentences.
        Returns: (prediction, probability)
        """
        inputs = tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        return bool(prediction), confidence

    # Test prediction
    test_sent1 = test_df.iloc[0]['sentence1']
    test_sent2 = test_df.iloc[0]['sentence2']
    true_label = test_df.iloc[0]['label']

    pred, conf = predict_authorship_change(test_sent1, test_sent2, model, tokenizer, device)

    print("Example prediction:")
    print(f"Sentence 1: {test_sent1[:100]}...")
    print(f"Sentence 2: {test_sent2[:100]}...")
    print(f"True label: {true_label}")
    print(f"Predicted: {pred} (confidence: {conf:.2%})")