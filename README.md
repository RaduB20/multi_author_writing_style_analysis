# Multi-Author Writing Style Analysis (In progress)

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
*TODO*

## Requirements
*TODO*
```bash
pip install -r requirements.txt
```

## Usage
*TODO*
```bash
python train.py --dataset easy
python evaluate.py --dataset easy --model checkpoints/best_model.pt
```

## Results
*TODO*

---
