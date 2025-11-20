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
if not os.path.exists(csv_path) or ALWAYS_PARSE:
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# We will try both with a normal topology and with a siamese topology

# https://huggingface.co/spaces/mteb/leaderboard
# BERT: DeBERTaV3, ModernBERT, NeoBERT, RoBERTa, BERT
# https://huggingface.co/infgrad/Jasper-Token-Compression-600M
# https://huggingface.co/Qwen/Qwen3-Embedding-8B
# https://huggingface.co/Qwen/Qwen3-Embedding-4B

# ----- Code cell 5 -----

CUSTOM_MODEL_NAME = 'custom-lightweight-transformer'
MODEL_NAME = CUSTOM_MODEL_NAME  # Change to 'prajjwal1/bert-mini', 'microsoft/deberta-v3-small', etc.

print("="*60)
print(f"Using {'CUSTOM' if MODEL_NAME == CUSTOM_MODEL_NAME else 'PRETRAINED'} MODEL: {MODEL_NAME}")
print("="*60)

# ----- Code cell 6 -----

# Designing a custom model so that the training is faster
class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, max_length=512, num_labels=2, dropout=0.1,
                 pad_token_id=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels)
        )
        
        self.d_model = d_model
        self.num_labels = num_labels
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'max_length': max_length,
            'num_labels': num_labels,
            'dropout': dropout
        }
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = min(input_ids.size(1), self.config['max_length'])
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids[:, :seq_len]) + self.position_embedding(pos_ids)
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        pooled = x[:, 0, :]
        logits = self.classifier(pooled)
        
        output = type('Output', (), {'logits': logits})()
        
        if labels is not None:
            output.loss = F.cross_entropy(logits, labels)
        
        return output
    
    def save_pretrained(self, path):
        """HuggingFace interface for saving the model"""
        
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid path: {path!r}. Path must be a non-empty string.")
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory '{path}': {e}")
        try:
            # Save model weights
            torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        except Exception as e:
            raise RuntimeError(f"Failed to save model weights to '{path}': {e}")
        try:
            with open(os.path.join(path, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to '{path}': {e}")
        print(f"Custom model saved to {path}")
    @classmethod
    def from_pretrained(cls, path):
        """HuggingFace interface for loading the model"""
        
        # Load config
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), weights_only=True))
        
        return model
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

# ----- Code cell 7 -----

from transformers import AutoTokenizer

MAX_LENGTH = 256  # Max token length for both sentences combined
NUM_LABELS = 2  # True and False (1 and 0)

if MODEL_NAME == CUSTOM_MODEL_NAME:
    
    TOKENIZER_NAME = 'gpt2'
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    OUTPUT_PATH = os.path.join('.', 'results', 'custom-transformer')
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default
    
    model = LightweightTransformer(
        vocab_size=len(tokenizer),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_length=MAX_LENGTH,
        num_labels=NUM_LABELS,
        dropout=0.25,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    print(f"Model size: ~{total_params * 4 / 1e6:.2f} MB (float32)")
else:
    from transformers import AutoModelForSequenceClassification
    
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    OUTPUT_PATH = os.path.join('.', 'results', MODEL_NAME.replace("/", "-"))
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, problem_type="single_label_classification"
    ).to(device)
    
    print(f"Model loaded with {model.num_parameters():,} parameters")

BEST_MODEL_PATH = os.path.join(OUTPUT_PATH, 'best_model')

# Token length analysis
print("\nAnalyzing token lengths...")
sample_indices = df.sample(min(1000, len(df)), random_state=RANDOM_SEED).index
token_lengths = pd.Series([
    len(tokenizer(df.loc[idx, 'sentence1'], df.loc[idx, 'sentence2'], truncation=False)['input_ids'])
    for idx in sample_indices
])

print(f"Token length statistics:")
print(f"  Mean: {token_lengths.mean():.0f}, Median: {token_lengths.median():.0f}")
print(f"  95th: {token_lengths.quantile(0.95):.0f}, 99th: {token_lengths.quantile(0.99):.0f}, Max: {token_lengths.max()}")
for length in [256, 384, 512]:
    print(f"  {(token_lengths > length).mean() * 100:.2f}% truncated at MAX_LENGTH={length}")

# ----- Code cell 8 -----

from torch.utils.data import Dataset

print("\nSetting up datasets for lazy tokenization...")

class LazyTokenizationDataset(Dataset):
    """
    Custom PyTorch Dataset that tokenizes data samples on-the-fly (lazily).
    This significantly reduces memory usage compared to eager tokenization.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        label = int(row['label']) # Convert boolean to 0 or 1

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt' # Return as PyTorch tensors
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label_tensor

train_dataset = LazyTokenizationDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = LazyTokenizationDataset(validation_df, tokenizer, MAX_LENGTH)
test_dataset = LazyTokenizationDataset(test_df, tokenizer, MAX_LENGTH)

NUM_WORKERS = 0 if os.name == 'nt' else 4  # 0 for Windows, 4 for Linux/Mac

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Lazy tokenization setup complete!")

# ----- Code cell 9 -----

def compute_metrics(all_preds, all_labels, all_probs):
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not compute AUC-ROC: {e}")
        auc_roc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

# ----- Markdown cell -----

# Training loop with early stopping

# ----- Code cell 10 -----

if __name__ == '__main__':
    
    # Adam optimizer with L2 regularization (weight_decay) to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,  # Peak learning rate
        total_steps=total_steps,
        pct_start=0.1,  # Warm-up: first 10% of training increases LR to max, then 90% decreases LR with cosine annealing (may drop below initial LR)
        anneal_strategy='cos'  # Cosine decreases LR smoothly, in contrast to 'linear'
    )
    
    print("\nStarting training...")
    print(f"Total training steps: {total_steps}")
    print("="*50)
    
    best_f1 = 0
    patience_counter = 0
    MAX_PATIENCE = 3  # Stop training after MAX_PATIENCE epochs without improvement
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_preds, train_labels_list = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            outputs.loss.backward()
            
            # Gradient clipping: prevents exploding gradients by capping them at 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {outputs.loss.item():.4f}")
        
        train_acc = accuracy_score(train_labels_list, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Metrics - Acc: {val_metrics['accuracy']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc_roc']:.4f}")
        print("-"*50)
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            model.save_pretrained(BEST_MODEL_PATH)
            print(f"✓ New best model saved to {BEST_MODEL_PATH} (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

    tokenizer.save_pretrained(BEST_MODEL_PATH)
    print(f"Tokenizer saved to {BEST_MODEL_PATH}")

# ----- Markdown cell -----

# The following cell should be able to run independently after training is complete, since it loads the best model from disk

# ----- Code cell 11 -----

    if MODEL_NAME == CUSTOM_MODEL_NAME:
        model = LightweightTransformer.from_pretrained(BEST_MODEL_PATH)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)

# ----- Code cell 12 -----

    print("\nEvaluating on validation set...")
    val_results = evaluate(model, val_loader, device)
    
    print("\nValidation Results:")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        print(f"  {key.replace('_', '-').title():10s}: {val_results[key]:.4f}")

# ----- Code cell 13 -----

    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device)
    
    print("\nTest Results:")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        print(f"  {key.replace('_', '-').title():10s}: {test_results[key]:.4f}")

# ----- Code cell 14 -----

# Inference on new data

def predict_authorship_change(sentence1: str, sentence2: str, model, tokenizer, device, max_length=256):
    """Predict authorship change between two sentences"""
    inputs = tokenizer(sentence1, sentence2, max_length=max_length, 
                      padding='max_length', truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    return bool(prediction), confidence

# Test prediction
test_sent1 = test_df.iloc[0]['sentence1']
test_sent2 = test_df.iloc[0]['sentence2']
true_label = test_df.iloc[0]['label']

pred, conf = predict_authorship_change(test_sent1, test_sent2, model, tokenizer, device, MAX_LENGTH)

print("\n" + "="*60)
print("Example prediction:")
print("="*60)
print(f"Sentence 1: {test_sent1[:100]}...")
print(f"Sentence 2: {test_sent2[:100]}...")
print(f"True label: {true_label}")
print(f"Predicted: {pred} (confidence: {conf:.2%})")
print(f"\nModel used: {MODEL_NAME}")