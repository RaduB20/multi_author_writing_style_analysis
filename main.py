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
    "load_rows": 10
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