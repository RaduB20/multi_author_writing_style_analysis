# Multi-Author Writing Style Analysis

## Overview
This project tackles the **style change detection task** for multi-author documents. The goal is to identify positions within a text where the writing style changes, indicating a potential switch in authorship. This is accomplished at the **sentence level** by analyzing each pair of consecutive sentences.

## Task Description
Given a multi-author document, detect all style change positions between consecutive sentences. This has practical applications in:
- Plagiarism detection (without comparison texts)
- Uncovering gift authorships
- Verifying claimed authorship
- Developing writing support technology

## Dataset
**Dataset source**: [PAN25 Multi-Author Writing Style Analysis](https://zenodo.org/records/14891299)

The project uses three difficulty levels, each controlling the relationship between topic and authorship changes:

- **Easy**: High topic diversity across sentences (topic can signal authorship changes)
- **Medium**: Low topic diversity (requires focus on stylistic features)
- **Hard**: All sentences share the same topic (pure style analysis)

Each dataset is split into:
- **Training set** (70%): With ground truth for model development
- **Validation set** (15%): With ground truth for model optimization
- **Test set** (15%): Without ground truth for final evaluation

## Key Constraints
- All documents are in English
- Documents may contain arbitrary numbers of style changes
- Style changes occur **only between sentences** (never within a sentence)
- Single sentences are always single-authored

## Approach

This project implements and compares several transformer-based approaches:

### Model Architectures

1. **Custom Lightweight Transformer**: A small transformer trained from scratch (~7M parameters) using GPT-2 tokenizer
   - Learned positional embeddings
   - Multi-head self-attention
   - Layer normalization and GELU activation
   - CLS token pooling for sequence representation

2. **Pretrained Models**: Fine-tuned HuggingFace models including:
   - `prajjwal1/bert-mini`: Compact BERT variant
   - `microsoft/deberta-v3-small`: Enhanced BERT with disentangled attention
   - `roberta-base`: Robustly optimized BERT approach

3. **Siamese Architecture**: Dual-encoder models that:
   - Encode each sentence separately using shared weights
   - Compare embeddings using multiple similarity methods:
     - **Concatenation**: `[emb1, emb2]`
     - **Absolute difference**: `|emb1 - emb2|`
     - **Element-wise multiplication**: `emb1 * emb2`
     - **Cosine similarity**: Angular alignment between embeddings

### Handling Class Imbalance

The dataset exhibits significant class imbalance (most sentence pairs are same-author). We address this through:

- **Weighted Random Sampling**: Oversamples minority class during training
- **Label Smoothing** (0.1): Regularization to prevent overconfident predictions
- **Data Augmentation**: Optional sentence swapping to increase effective dataset size

### Training Strategy

- **Optimizer**: AdamW with weight decay (0.1) for L2 regularization
- **Learning Rate Schedule**: OneCycleLR with cosine annealing
  - Warmup phase (10% of training)
  - Peak learning rate based on model type
  - Gradual decay to minimum
- **Encoder Freezing**: Progressive fine-tuning option
  - Initially freeze pretrained encoder
  - Unfreeze after specified fraction of epochs
  - Prevents catastrophic forgetting of pretrained knowledge
- **Gradient Clipping**: Max norm of 1.0 for training stability
- **Early Stopping**: Patience of 3 epochs based on validation F1 score

### Evaluation Metrics

- **F1 Score** (primary metric): Harmonic mean of precision and recall
- **Accuracy**: Overall correctness
- **Precision**: Fraction of predicted style changes that are correct
- **Recall**: Fraction of actual style changes detected
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Requirements

### Python Version
- Python 3.12

### Installation

```bash
pip install -r requirements.txt
```

### For PyTorch with CUDA Support (Windows)
If using Windows with CUDA 12.1:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## Usage

### Running the Complete Pipeline

The main analysis is contained in `main.ipynb`. Open it in Jupyter:

```bash
jupyter notebook main.ipynb
```

The notebook includes:
1. **Data Loading & EDA**: Automatic download, preprocessing, and exploratory analysis
2. **Model Architecture**: Definition of encoders, classification heads, and full models
3. **Training**: Training loop with class imbalance handling and regularization
4. **Single Model Experiment**: Train and visualize one model's performance
5. **Model Comparison**: Train multiple models and compare test results
6. **Inference**: Interactive predictions on new sentence pairs

### Quick Start Examples

#### Train a Single Model

```python
from main import get_model_config, train_model, load_model_from_config

config = get_model_config('microsoft/deberta-v3-small', device)

model, result = train_model(
    config=config,
    train_df=train_df,
    val_df=validation_df
)
```

#### Compare Multiple Models

```python
comparison_df = compare_models(
    model_names=[
        'custom-lightweight-transformer',
        'prajjwal1/bert-mini',
        'microsoft/deberta-v3-small',
        'siamese-roberta-base'
    ],
    train_df=train_df,
    val_df=validation_df,
    test_df=test_df
)
```

#### Make Predictions

```python
# Load trained model
config = get_model_config('microsoft/deberta-v3-small', device)
model = load_model_from_config(config)
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

# Predict on new sentence pair
sentence1 = "The empirical analysis demonstrates a statistically significant correlation."
sentence2 = "lol yeah that's pretty cool i guess, dunno why anyone would care tho"

prediction, confidence = predict_authorship_change(
    sentence1, sentence2, model, tokenizer, device, max_length=128
)

print(f"Different authors: {prediction} (confidence: {confidence:.2%})")
```

## Results

### Performance on Easy Dataset

Best model: **microsoft/deberta-v3-small**

| Model | Test F1 | Test Accuracy | Test Precision | Test Recall | Test AUC-ROC |
|-------|---------|---------------|----------------|-------------|--------------|
| microsoft/deberta-v3-small | 0.923 | 0.973 | 0.916 | 0.930 | 0.970 |
| siamese-prajjwal1/bert-mini | 0.890 | 0.960 | 0.848 | 0.936 | 0.988 |
| roberta-base | 0.876 | 0.952 | 0.794 | 0.977 | 0.992 |
| siamese-roberta-base | 0.871 | 0.951 | 0.806 | 0.949 | 0.984 |

### Comparison with State-of-the-Art (PAN 2025)

| Team | Approach | Easy | Medium | Hard | Average F1 |
|------|----------|------|--------|------|------------|
| xxsu-team | SCL-DeBERTa | 0.955 | 0.825 | 0.829 | 0.870 |
| stylospies | Graph/Structural Features | 0.959 | 0.786 | 0.791 | 0.845 |
| TMU | Ensemble LaBSE/Siamese BiLSTM | 0.950 | 0.792 | 0.792 | 0.845 |
| better_call_claude | SSPC (BiLSTM/PLM) | 0.929 | 0.815 | 0.731 | 0.825 |
| cornell-1 | Ensembled-BertStyleNN | 0.909 | 0.793 | 0.698 | 0.800 |
| OpenFact | Punctuation-Guided Pretraining | 0.919 | 0.771 | 0.752 | 0.814 |
| **Arduo (Ours)** | **microsoft/deberta-v3-small** | **0.922** | **0.715** | **0.694** | **0.777** |

### Key Findings

1. **Class Imbalance**: Weighted sampling proved more effective than weighted loss for handling the 4:1 imbalance ratio
2. **Siamese Architecture**: Mixed results across different base models; requires further investigation with consistent hyperparameters
3. **Custom vs. Pretrained**: Pretrained models significantly outperform custom architectures, highlighting the importance of language understanding from pretraining
4. **Regularization**: Dropout (0.33), label smoothing (0.1), and gradient clipping were critical for preventing overfitting
5. **Performance Gap**: While competitive on easy tasks, our approach lags on medium/hard difficulties, suggesting the need for domain-specific features or ensemble methods

## Project Structure

```
.
├── main.ipynb              # Main analysis notebook
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory (auto-created)
│   ├── easy/               # Easy difficulty dataset
│   ├── medium/             # Medium difficulty dataset
│   ├── hard/               # Hard difficulty dataset
│   └── loaded_data.csv     # Cached processed data
└── results/                # Trained models directory (auto-created)
    ├── custom-lightweight-transformer/
    ├── microsoft-deberta-v3-small/
    └── ...
```

## Future Work

- **Hyperparameter Optimization**: Grid search or Bayesian optimization for better parameter tuning
- **Contrastive Learning**: Implement supervised contrastive learning (SCL) similar to top-performing teams
- **Domain-Specific Features**: Incorporate linguistic features (punctuation patterns, sentence structure, vocabulary richness)
- **Ensemble Methods**: Combine predictions from multiple models for improved robustness
- **Data Augmentation**: Back-translation, paraphrasing, or SMOTE on embeddings
- **Focal Loss**: Alternative loss function specifically designed for imbalanced datasets
- **Larger Models**: Fine-tune DeBERTa-large or RoBERTa-large for potentially better performance

## Authors

- Juan Arturo Abaurrea Calafell
- Radu-Andrei Bourceanu

---

For detailed implementation and experiments, see `main.ipynb`.