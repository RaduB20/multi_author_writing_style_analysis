# Later we will turn this into a jupyter notebook

# Multi Author Writing Style Analysis
# Authors: Juan Arturo Abaurrea Calafell, Radu-Andrei Bourceanu

# 3 parts: EDA, Training, Evaluation

# Exploratory Data Analysis (EDA)

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