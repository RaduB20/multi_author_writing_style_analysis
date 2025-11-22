# Later we will turn this into a jupyter notebook

# Multi Author Writing Style Analysis
# Authors: Juan Arturo Abaurrea Calafell, Radu-Andrei Bourceanu

# 3 parts: EDA, Training, Evaluation

# Part 1: Exploratory Data Analysis (EDA)

# ----- Code cell 1 -----

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # to avoid potential issues with macOS and some Windows setups
import requests
import zipfile
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

RANDOM_SEED = 42
NUM_WORKERS = 0 if os.name == 'nt' else 4  # 0 for Windows, 4 for Linux/Mac
ALWAYS_PARSE = False  # Set to True to always re-parse the data
ALWAYS_TRAIN = False  # Set to True to always re-train the model

# Class imbalance parameters
UPSAMPLE_UNDERREPRESENTED_CLASS = False
WEIGHTED_LOSS = False
WEIGHTED_SAMPLER = True
LABEL_SMOOTHING = True

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

DATA_LOADING_PARAMETERS = {
    "difficulty_levels": DIFFICULTY_LEVELS[0],
    "swap_different_author_sentences": UPSAMPLE_UNDERREPRESENTED_CLASS,
    "load_problems": None  # Set to None to load all the data
}

def load_data(data_dir: str, difficulty_levels: list[str] | str, use_all_possible_pairs: bool = False, swap_same_author_sentences: bool = False, swap_different_author_sentences: bool = False, load_problems: int = None) -> pd.DataFrame:
    """
    Loads data from the specified directory and difficulty levels and returns it as a single dataframe.

    Parameters:
        data_dir: Directory containing the data.
        difficulty_levels: List of difficulty levels to load (e.g., ['easy', 'medium']).
        use_all_possible_pairs: For sentences by the same author, creates all possible pairs.
        load_problems: Maximum number of problem files to load per split and difficulty level. If None, loads all.
        swap_same_author_sentences, swap_different_author_sentences: If True, creates additional rows with sentence1 and sentence2 swapped for the respective case.
    Returns:
        pd.DataFrame: DataFrame with columns 'sentence1', 'sentence2', 'label'.
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
            
            for idx, problem_file in enumerate(problem_files): # Iterate over problem files
                if load_problems is not None and idx >= load_problems:
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
                    authors = truth_data.get('authors', 0)
                    changes = truth_data.get('changes', [])
                
                if len(sentences) != len(changes) + 1 or authors < 1:
                    skipped_count += 1
                    continue

                if not use_all_possible_pairs:
                    # Create pairs only between 2 consecutive sentences
                    for i in range(len(sentences) - 1):
                        rows.append({
                            'sentence1': sentences[i],
                            'sentence2': sentences[i + 1],
                            'label': changes[i] != 0  # Storing as boolean (label: True if changed, False otherwise)
                        })
                else:
                    # If there are exactly 2 authors, use optimized approach
                    if authors == 2:
                        # Assign sentences to authors
                        author_sentences = [[], []]  # [author_0_sentences, author_1_sentences]
                        current_author = 0
                        
                        for i in range(len(sentences)):
                            author_sentences[current_author].append(sentences[i])
                            if i < len(changes) and changes[i] != 0:
                                current_author = 1 - current_author  # Switch between 0 and 1
                        
                        # Create all pairs within each author group (label=False)
                        for author_group in author_sentences:
                            for i in range(len(author_group)):
                                for j in range(i + 1, len(author_group)):
                                    rows.append({
                                        'sentence1': author_group[i],
                                        'sentence2': author_group[j],
                                        'label': False  # Same author
                                    })
                        
                        # Create all pairs across authors (label=True)
                        for sent_a in author_sentences[0]:
                            for sent_b in author_sentences[1]:
                                rows.append({
                                    'sentence1': sent_a,
                                    'sentence2': sent_b,
                                    'label': True  # Different authors
                                })
                    else:
                        # For more than 2 authors, use the sequential approach
                        i = 0
                        while i < len(sentences): # Iterate over sentences in the current problem file
                            # Find the extent of the current author group
                            j = i
                            while j < len(changes) and changes[j] == 0:
                                j += 1
                            
                            # sentences[i:j+1] are all from the same author
                            group_size = j - i + 1
                            
                            if group_size > 1:
                                # Create all pairs within this group
                                for k in range(i, j + 1):
                                    for l in range(k + 1, j + 1):
                                        rows.append({
                                            'sentence1': sentences[k],
                                            'sentence2': sentences[l],
                                            'label': False  # Same author
                                        })
                            
                            # If there is a change after position j, add the cross-boundary pair
                            # This will only be false at the end of the file
                            if j < len(changes) and changes[j] != 0:
                                rows.append({
                                    'sentence1': sentences[j],
                                    'sentence2': sentences[j + 1],
                                    'label': True  # Different authors
                                })
                            
                            # Next author group
                            i = j + 1
    
    df = pd.DataFrame(rows)

    df['sentence1'] = df['sentence1'].astype('string')
    df['sentence2'] = df['sentence2'].astype('string')
    df['label'] = df['label'].astype('boolean')

    print(f"Total documents skipped due to mismatches: {skipped_count}")
    
    if swap_same_author_sentences:
        df_same = df[df['label'] == False].copy()
        df_same['sentence1'], df_same['sentence2'] = df_same['sentence2'], df_same['sentence1']
        rows_added = len(df_same)
        df = pd.concat([df, df_same], ignore_index=True)
        print(f"Added {rows_added} swapped same-author sentence pairs to the dataset")

    if swap_different_author_sentences:
        df_diff = df[df['label'] == True].copy()
        df_diff['sentence1'], df_diff['sentence2'] = df_diff['sentence2'], df_diff['sentence1']
        rows_added = len(df_diff)
        df = pd.concat([df, df_diff], ignore_index=True)
        print(f"Added {rows_added} swapped different-author sentence pairs to the dataset")

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

# Part 2: Training the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# https://huggingface.co/spaces/mteb/leaderboard
# BERT: DeBERTaV3, ModernBERT, NeoBERT, RoBERTa, BERT
# https://huggingface.co/infgrad/Jasper-Token-Compression-600M
# https://huggingface.co/Qwen/Qwen3-Embedding-8B
# https://huggingface.co/Qwen/Qwen3-Embedding-4B

# ----- Code cell 4 -----

CUSTOM_MODEL_NAME = 'custom-lightweight-transformer'
SIAMESE_PREFIX = 'siamese-'  # do not change

# ----- Code cell 5 -----

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders (custom and pretrained)"""
    
    @abstractmethod
    def encode(self, input_ids, attention_mask=None):
        """Encode input and return pooled representation"""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the dimension of the encoder output"""
        pass
    
    @abstractmethod
    def save_pretrained(self, path: str):
        """Save encoder to disk"""
        pass
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str):
        """Load encoder from disk"""
        pass

class LightweightTransformerEncoder(BaseEncoder):
    """Custom lightweight transformer that only encodes (no classification head)"""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, max_length, dropout, pad_token_id):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        self.d_model = d_model
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'max_length': max_length,
            'dropout': dropout,
            'pad_token_id': pad_token_id
        }
    
    def encode(self, input_ids, attention_mask=None):
        seq_len = min(input_ids.size(1), self.config['max_length'])
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids[:, :seq_len]) + self.position_embedding(pos_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        return x[:, 0, :]  # CLS token
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), weights_only=True))
        return model
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

class PretrainedEncoder(BaseEncoder):
    """Wrapper for HuggingFace pretrained models"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self._output_dim = self.model.config.hidden_size
    
    def encode(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]  # CLS token
    
    def get_output_dim(self) -> int:
        return self._output_dim
    
    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        
        with open(os.path.join(path, 'encoder_info.json'), 'w') as f:
            json.dump({'model_name': self.model_name}, f)
    
    @classmethod
    def from_pretrained(cls, path: str):
        # Create instance without loading model in __init__
        instance = object.__new__(cls)
        nn.Module.__init__(instance)
        instance.model = AutoModel.from_pretrained(path)
        instance._output_dim = instance.model.config.hidden_size
        with open(os.path.join(path, 'encoder_info.json'), 'r') as f:
            info = json.load(f)
        instance.model_name = info['model_name']
        return instance
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

class ClassificationHead(nn.Module):
    """Simple classification head"""
    
    def __init__(self, input_dim: int, dropout: float, num_labels: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_labels)
        )
        self.config = {'input_dim': input_dim, 'num_labels': num_labels, 'dropout': dropout}
    
    def forward(self, x):
        return self.classifier(x)
    
    def save_pretrained(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, 'classifier.bin'))
        with open(os.path.join(path, 'classifier_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str):
        with open(os.path.join(path, 'classifier_config.json'), 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'classifier.bin'), weights_only=True))
        return model

class AuthorshipModel(nn.Module):
    """
    Model for authorship verification.
    Can operate in standard mode (concatenated input) or siamese mode (separate encoding).
    """
    
    def __init__(self, encoder: BaseEncoder, dropout: float, num_labels: int = 2,
                 siamese: bool = False, similarity_method: str = 'concat_diff_mult'):
        super().__init__()
        
        self.encoder = encoder
        self.siamese = siamese
        self.similarity_method = similarity_method
        self.num_labels = num_labels
        
        encoder_dim = encoder.get_output_dim()
        
        if siamese:
            # Classifier input depends on how we combine the two embeddings
            if similarity_method == 'concat':
                classifier_input = encoder_dim * 2
            elif similarity_method == 'concat_diff':
                classifier_input = encoder_dim * 3
            elif similarity_method == 'concat_diff_mult':
                classifier_input = encoder_dim * 4
            else:
                raise ValueError(f"Unknown similarity method: {similarity_method}")
        else:
            classifier_input = encoder_dim
        
        self.classifier = ClassificationHead(classifier_input, num_labels=num_labels, dropout=dropout)
        
        self.config = {
            'num_labels': num_labels,
            'dropout': dropout,
            'siamese': siamese,
            'similarity_method': similarity_method
        }
    
    def forward(self, input_ids1, attention_mask1, input_ids2=None, attention_mask2=None, labels=None, class_weights=None, label_smoothing=0.0):
        """
        Forward pass.
        - Standard mode: input_ids1/attention_mask1 contain concatenated sentences
        - Siamese mode: input_ids1/2 and attention_mask1/2 contain separate sentences
        """
        if self.siamese:
            if input_ids2 is None or attention_mask2 is None:
                raise ValueError("Siamese mode requires input_ids2 and attention_mask2")
            emb1 = self.encoder.encode(input_ids1, attention_mask1)
            emb2 = self.encoder.encode(input_ids2, attention_mask2)
            
            if self.similarity_method == 'concat':
                combined = torch.cat([emb1, emb2], dim=-1)
            elif self.similarity_method == 'concat_diff':
                diff = torch.abs(emb1 - emb2)
                combined = torch.cat([emb1, emb2, diff], dim=-1)
            elif self.similarity_method == 'concat_diff_mult':
                diff = torch.abs(emb1 - emb2)
                mult = emb1 * emb2
                combined = torch.cat([emb1, emb2, diff, mult], dim=-1)
            
            logits = self.classifier(combined)
        else:
            emb = self.encoder.encode(input_ids1, attention_mask1)
            logits = self.classifier(emb)
        
        output = type('Output', (), {'logits': logits})()
        
        if labels is not None:
            # We add label smoothing for better generalization since the dataset is very imbalanced
            output.loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
        
        return output
    
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.classifier.save_pretrained(path)
        with open(os.path.join(path, 'model_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, encoder_class: type):
        with open(os.path.join(path, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        encoder = encoder_class.from_pretrained(os.path.join(path, 'encoder'))
        model = cls(encoder, **config)
        model.classifier = ClassificationHead.from_pretrained(path)
        return model
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HuggingFaceModelWrapper(nn.Module):
    """Wrapper around HuggingFace models with custom classification head"""
    
    def __init__(self, model_name: str, dropout: float, num_labels: int = 2):
        super().__init__()
        # Load only the base model, not the classification head
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = ClassificationHead(
            input_dim=self.encoder.config.hidden_size,
            dropout=dropout,
            num_labels=num_labels
        )
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout = dropout
        self.siamese = False
    
    def forward(self, input_ids1, attention_mask1, input_ids2=None, attention_mask2=None, labels=None, class_weights=None, label_smoothing=0.0):
        outputs = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1)
        
        # Use pooler_output if available, otherwise CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled)
        
        output = type('Output', (), {'logits': logits})()
        
        if labels is not None:
            # We add label smoothing for better generalization since the dataset is very imbalanced
            output.loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
        
        return output
        
    def save_pretrained(self, path: str):
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.classifier.save_pretrained(path)
        with open(os.path.join(path, 'wrapper_info.json'), 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'dropout': self.dropout
            }, f)
    
    @classmethod
    def from_pretrained(cls, path: str):
        with open(os.path.join(path, 'wrapper_info.json'), 'r') as f:
            info = json.load(f)
        instance = object.__new__(cls)
        nn.Module.__init__(instance)
        instance.encoder = AutoModel.from_pretrained(os.path.join(path, 'encoder'))
        instance.classifier = ClassificationHead.from_pretrained(path)
        instance.model_name = info['model_name']
        instance.num_labels = info['num_labels']
        instance.dropout = info.get('dropout', 0.1)
        instance.siamese = False
        return instance
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

# ----- Code cell 6 -----

# Lazy dataset
class StandardDataset(Dataset):
    """Dataset that tokenizes sentence pairs together (for non-siamese models)"""
    
    def __init__(self, dataframe, tokenizer, max_length):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['sentence1'], row['sentence2'],
            padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(int(row['label']), dtype=torch.long)
        )

# Lazy dataset
class SiameseDataset(Dataset):
    """Dataset that tokenizes each sentence separately (for siamese models)"""
    
    def __init__(self, dataframe, tokenizer, max_length):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc1 = self.tokenizer(row['sentence1'], padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(row['sentence2'], padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')
        return (
            enc1['input_ids'].squeeze(0), enc1['attention_mask'].squeeze(0),
            enc2['input_ids'].squeeze(0), enc2['attention_mask'].squeeze(0),
            torch.tensor(int(row['label']), dtype=torch.long)
        )


# ----- Code cell 7 -----

def get_model_size_mb(model: nn.Module, assume_uniform_dtype: bool = True) -> float:
    """Calculate actual model size in MB based on parameter dtypes"""
    params = list(model.parameters())
    if not params:
        return 0.0
    
    if assume_uniform_dtype:
        return (model.num_parameters() * params[0].element_size()) / 1e6
    
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / 1e6

@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_name: str
    model: nn.Module
    device: torch.device
    tokenizer: AutoTokenizer
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    model_path: str
    is_siamese: bool
    is_custom: bool
    dataset_class: type
    encoder_class: type  # For loading


def get_model_config(
    model_name: str,
    device: torch.device,
    max_length: int = 256,
    num_labels: int = 2,
    # Custom model parameters
    custom_d_model: int = 128,
    custom_nhead: int = 4,
    custom_num_layers: int = 2,
    custom_dim_feedforward: int = 256,
    dropout: float = 0.25,
    # Siamese parameters
    siamese_similarity_method: str = 'concat_diff_mult',
    # Training parameters
    custom_batch_size: int = 64,
    custom_learning_rate: float = 1e-3,
    custom_num_epochs: int = 10,
    pretrained_batch_size: int = 16,
    pretrained_learning_rate: float = 2e-5,
    pretrained_num_epochs: int = 5,
) -> ModelConfig:
    """
    Creates model configuration based on model name.
    
    Supported formats:
        - 'custom-lightweight-transformer': Custom transformer
        - 'siamese-custom-lightweight-transformer': Siamese custom transformer  
        - 'prajjwal1/bert-mini': Pretrained HuggingFace model
        - 'siamese-prajjwal1/bert-mini': Siamese pretrained model
    
    Parameters:
        model_name: Name of the model
        device: Device to use
        max_length: Maximum sequence length
        num_labels: Number of output labels
        custom_*: Parameters for custom model
        siamese_similarity_method: Method to combine embeddings in siamese mode ('concat', 'concat_diff', 'concat_diff_mult')
        custom_*, pretrained_*: Training parameters for custom and pretrained models

    Returns:
        ModelConfig object
    """
    is_siamese = model_name.startswith(SIAMESE_PREFIX)
    base_model_name = model_name[len(SIAMESE_PREFIX):] if is_siamese else model_name
    is_custom = base_model_name == CUSTOM_MODEL_NAME
    
    model_path = os.path.join('.', 'results', model_name.replace("/", "-"))
    
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Type: {'Siamese ' if is_siamese else ''}{'Custom' if is_custom else 'Pretrained'}")
    print("=" * 60)
    
    if is_custom:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        encoder = LightweightTransformerEncoder(
            vocab_size=len(tokenizer),
            d_model=custom_d_model,
            nhead=custom_nhead,
            num_layers=custom_num_layers,
            dim_feedforward=custom_dim_feedforward,
            max_length=max_length,
            dropout=dropout,
            pad_token_id=tokenizer.pad_token_id
        )
        
        model = AuthorshipModel(
            encoder=encoder,
            num_labels=num_labels,
            dropout=dropout,
            siamese=is_siamese,
            similarity_method=siamese_similarity_method
        ).to(device)
        
        batch_size, learning_rate, num_epochs = custom_batch_size, custom_learning_rate, custom_num_epochs
        encoder_class = LightweightTransformerEncoder
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if is_siamese:
            encoder = PretrainedEncoder(base_model_name)
            model = AuthorshipModel(
                encoder=encoder,
                num_labels=num_labels,
                dropout=dropout,
                siamese=True,
                similarity_method=siamese_similarity_method
            ).to(device)
            encoder_class = PretrainedEncoder
        else:
            model = HuggingFaceModelWrapper(base_model_name, num_labels=num_labels, dropout=dropout).to(device)
            encoder_class = None  # Not used for HuggingFace wrapper
        
        batch_size, learning_rate, num_epochs = pretrained_batch_size, pretrained_learning_rate, pretrained_num_epochs
    
    dataset_class = SiameseDataset if is_siamese else StandardDataset
    
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Size: ~{get_model_size_mb(model):.2f} MB")
    
    return ModelConfig(
        model_name=model_name,
        model=model,
        device=device,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_path=model_path,
        is_siamese=is_siamese,
        is_custom=is_custom,
        dataset_class=dataset_class,
        encoder_class=encoder_class
    )


def load_model_from_config(config: ModelConfig, move_to_device: bool = True) -> nn.Module:
    """Load a model from disk using its config"""
    path = config.model_path
    
    if config.is_custom or config.is_siamese:
        model = AuthorshipModel.from_pretrained(path, config.encoder_class)
    else:
        model = HuggingFaceModelWrapper.from_pretrained(path)
    
    if move_to_device:
        model = model.to(config.device)
    return model


# ----- Code cell 8 -----

def compute_metrics(all_preds, all_labels, all_probs):
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not compute AUC-ROC ({type(e).__name__}): {e}")
        auc_roc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def evaluate(model, dataloader, device, is_siamese=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = [b.to(device) for b in batch]
            
            if is_siamese:
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2, labels=labels)
            else:
                input_ids, attention_mask, labels = batch
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


# ----- Code cell 9 -----

@dataclass
class TrainingResult:
    """Results from training a model"""
    best_f1: float
    best_epoch: int
    final_epoch: int
    training_history: list = field(default_factory=list)


def train_model(
    config: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_patience: int = 3,
    weight_decay: float = 0.1
) -> tuple[nn.Module, TrainingResult]:
    """
    Train a model and return results.
    
    Parameters:
        config: ModelConfig object containing model configuration
        train_df, val_df: DataFrames with 'sentence1', 'sentence2', 'label' columns
        max_patience: Early stopping patience
        weight_decay: L2 regularization
    
    Returns:
        tuple[nn.Module, TrainingResult]: The best model and training results including best F1, best epoch, final epoch, and training history
    """
    
    model = config.model
    best_model = model
    tokenizer = config.tokenizer
    
    labels = train_df['label'].astype(int).values
    class_counts = np.bincount(labels)  # [count_false, count_true]

    # For weighted loss function
    if WEIGHTED_LOSS:
        class_weights = torch.tensor(len(labels) / (2 * class_counts), dtype=torch.float32).to(config.device)
    else:
        class_weights = None
    
    train_dataset = config.dataset_class(train_df, tokenizer, config.max_length)
    val_dataset = config.dataset_class(val_df, tokenizer, config.max_length)

    # For weighted sampler to balance classes in each batch
    if WEIGHTED_SAMPLER:
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    label_smoothing = 0.1 if LABEL_SMOOTHING else 0.0
    
    print(f"\nTraining {config.model_name}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Batch size: {config.batch_size}, LR: {config.learning_rate}, Epochs: {config.num_epochs}")
    print(f"Total steps: {total_steps}")
    print("-" * 50)

    os.makedirs(config.model_path, exist_ok=True)
    model.save_pretrained(config.model_path)
    tokenizer.save_pretrained(config.model_path)
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels_list = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = [b.to(config.device) for b in batch]
            
            if config.is_siamese:
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2, labels=labels, class_weights=class_weights, label_smoothing=label_smoothing)
            else:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights, label_smoothing=label_smoothing)
            
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {outputs.loss.item():.4f}")
        
        train_acc = accuracy_score(train_labels_list, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, config.device, is_siamese=config.is_siamese)
        
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        training_history.append(epoch_record)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} stats")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
        
        # Early stopping check
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0
            
            model.save_pretrained(config.model_path)
            best_model = model
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        print()
    
    print("\n" + "=" * 50)
    print(f"Training complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {config.model_path}")
    print("=" * 50)
    
    return (
        best_model,
        TrainingResult(
            best_f1=best_f1,
            best_epoch=best_epoch,
            final_epoch=epoch + 1,
            training_history=training_history
        )
    )


# ----- Code cell 10 -----

def compare_models(
    model_names: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_length: int = 256,
    **kwargs
) -> pd.DataFrame:
    """
    Train and compare multiple models.
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for i, model_name in enumerate(model_names):
        print(f"\n{'#' * 70}")
        print(f"# Model {i+1}/{len(model_names)}: {model_name}")
        print(f"{'#' * 70}\n")
        
        try:
            config = get_model_config(model_name, device, max_length=max_length, **kwargs)

            # if the model has already been trained, load it
            try:
                if (not ALWAYS_TRAIN) and os.path.exists(config.model_path):
                    print(f"\nLoading trained model from {config.model_path}...")
                    model = load_model_from_config(config)
                else:
                    raise FileNotFoundError
            except Exception as e:
                if e is FileNotFoundError:
                    print(f"The model was not found at {config.model_path}.")
                else:
                    print(f"Could not load model: {e}")
                print(f"\nTraining model {model_name} from scratch...")
                model, _ = train_model(
                    config=config,
                    train_df=train_df,
                    val_df=val_df
                )

            # Evaluate on test set
            test_dataset = config.dataset_class(test_df, config.tokenizer, config.max_length)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS)
            test_metrics = evaluate(model, test_loader, config.device, is_siamese=config.is_siamese)
            
            results.append({
                'model': model_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'test_auc_roc': test_metrics['auc_roc'],
                'model_path': config.model_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            
            results.append({
                'model': model_name,
                'status': 'failed',
                'error': str(e)
            })
    
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    successful = comparison_df[comparison_df['status'] == 'success']
    if len(successful) > 0:
        print("\nTest Set Results (sorted by F1):")
        display_cols = ['model', 'test_f1', 'test_accuracy', 'test_precision', 'test_recall', 'test_auc_roc']
        print(successful[display_cols].sort_values('test_f1', ascending=False).to_string(index=False))
        
        best_model = successful.loc[successful['test_f1'].idxmax()]
        print(f"\nBest model: {best_model['model']} (Test F1: {best_model['test_f1']:.4f})")
    
    failed = comparison_df[comparison_df['status'] == 'failed']
    if len(failed) > 0:
        print(f"\nFailed models: {', '.join(failed['model'].tolist())}")

    return comparison_df

# ----- Code cell 11 -----

def predict_authorship_change(
    sentence1: str,
    sentence2: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 256,
    is_siamese: bool = False
) -> tuple[bool, float]:
    """
    Predict whether two sentences were written by different authors.
    
    Returns:
        (prediction, confidence): True if different authors, with confidence score
    """
    model.eval()
    
    if is_siamese:
        enc1 = tokenizer(sentence1, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
        enc2 = tokenizer(sentence2, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(
                enc1['input_ids'].to(device), enc1['attention_mask'].to(device),
                enc2['input_ids'].to(device), enc2['attention_mask'].to(device)
            )
    else:
        enc = tokenizer(sentence1, sentence2, padding='max_length', truncation=True,
                       max_length=max_length, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    
    probs = F.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()
    
    return bool(prediction), confidence


# ----- Code cell 12 -----

TRAIN_RATIO = 0.7
VALIDATION_TEST_RATIO = 0.5

train_df, temp_df = train_test_split(df, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, stratify=df['label'])
validation_df, test_df = train_test_split(temp_df, train_size=VALIDATION_TEST_RATIO, random_state=RANDOM_SEED, stratify=temp_df['label'])

print(f"Training set size: {len(train_df)}. Authorship change rate: {train_df['label'].mean():.2%}")
print(f"Validation set size: {len(validation_df)}. Authorship change rate: {validation_df['label'].mean():.2%}")
print(f"Test set size: {len(test_df)}. Authorship change rate: {test_df['label'].mean():.2%}")


# ----- Markdown cell -----

# Training one model

# ----- Code cell 13 -----

if __name__ == '__main__':
    
    # Some options:
    #
    # CUSTOM_MODEL_NAME
    # f'{SIAMESE_PREFIX}{CUSTOM_MODEL_NAME}'
    # 'prajjwal1/bert-mini'
    # f'{SIAMESE_PREFIX}prajjwal1/bert-mini'
    
    siamese_mode = False
    model_name = f'{CUSTOM_MODEL_NAME}'

    config = get_model_config(model_name, device)

    # if the model has already been trained, load it
    try:
        if (not ALWAYS_TRAIN) and os.path.exists(config.model_path):
            print(f"\nLoading trained model from {config.model_path}...")
            model = load_model_from_config(config)
        else:
            raise FileNotFoundError
    except Exception as e:
        if e is FileNotFoundError:
            print(f"The model was not found at {config.model_path}.")
        else:
            print(f"Could not load model: {e}")
        print(f"\nTraining model {model_name} from scratch...")
        model, result = train_model(
            config=config,
            train_df=train_df,
            val_df=validation_df
        )
        
        history_df = pd.DataFrame(result.training_history)
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['train_loss'], label='Train Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    
    print("\nEvaluating model on test set...")
    test_dataset = config.dataset_class(test_df, config.tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_metrics = evaluate(model, test_loader, device, is_siamese=config.is_siamese)
    
    print("\nTest Results:")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        print(f"  {key.replace('_', '-').title():10s}: {test_metrics[key]:.4f}")
    
# ----- Code cell 14 -----
    
    """
    MODELS_TO_COMPARE = [
        CUSTOM_MODEL_NAME,
        f'{SIAMESE_PREFIX}{CUSTOM_MODEL_NAME}',
        'prajjwal1/bert-mini',
        f'{SIAMESE_PREFIX}prajjwal1/bert-mini',
    ]
    
    comparison_df = compare_models(
        model_names=MODELS_TO_COMPARE,
        train_df=train_df,
        val_df=validation_df,
        test_df=test_df
    )
    
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("\nComparison saved to model_comparison_results.csv")
    """

# ----- Markdown cell -----

# Testing a trained model for prediction on specific sentences

# ----- Code cell 15 -----
    
    model_name = CUSTOM_MODEL_NAME
    config = get_model_config(model_name, device)
    model = load_model_from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    test_sent1 = test_df.iloc[0]['sentence1']
    test_sent2 = test_df.iloc[0]['sentence2']
    true_label = test_df.iloc[0]['label']
    
    pred, conf = predict_authorship_change(
        test_sent1, test_sent2, model, tokenizer, device,
        max_length=256, is_siamese=config.is_siamese
    )
    
    print("\n" + "=" * 60)
    print("Example prediction:")
    print("=" * 60)
    print(f"Sentence 1: {test_sent1[:100]}...")
    print(f"Sentence 2: {test_sent2[:100]}...")
    print(f"True label: {true_label}")
    print(f"Predicted: {pred} (confidence: {conf:.2%})")
    print(f"Model: {model_name}")

# ----- Markdown cell -----

# Conclusion

# If we were to continue improving this project, some possible directions include:
# - Adding SMOTE or other data augmentation techniques to address class imbalance.
# - Using Focal Loss or other loss functions tailored for imbalanced datasets.
