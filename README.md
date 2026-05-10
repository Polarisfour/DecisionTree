# Decision Tree Classifier
This project implements a decision tree classifier from scratch in Python.  
It supports two common attribute selection metrics—Gain Ratio and Gini Index—and includes a simple cross-validation routine to evaluate model performance.

---

## Overview

I built this project as part of an algorithms course to learn how decision trees work under the hood.  
The code constructs a tree using either Gain Ratio or Gini Index, classifies examples, and calculates F1 scores over different validation sets.

Key features:
- Implements decision tree induction with Gain Ratio and Gini Index
- Handles missing data by replacing with median values
- Splits continuous attributes into high/low bins automatically
- Evaluates using cross-validation and F1-score on a test dataset

---

## Technologies Used
- Python 3
- Built-in libraries: `math`, `copy`
- Custom classes for `Example`, `Node`, and `Attribute`

---

## Files
- `decisiontree.py` – main code with decision tree implementation and evaluation
- `training.data` – training dataset (expected format: CSV with features and +/– class labels)
- `test.data` – test dataset (same format)

---

## How It Works

1. **Preprocessing**  
   - Reads training and test data from `.data` files.  
   - Converts continuous attributes into discrete high/low categories using median split points.  
   - Handles missing values by filling with medians.

2. **Training**  
   - Constructs two decision trees: one using Gain Ratio, one using Gini Index.  
   - Recursively splits examples based on attribute importance.  

3. **Evaluation**  
   - Performs 10-fold style cross-validation to compute F1 scores for each method.  
   - Outputs best validation ranges and final F1 scores on test data.

---

## Installation

Clone the repository and install Python 3 if you don’t already have it:

```bash
git clone https://github.com/yourusername/decision-tree-classifier.git
cd decision-tree-classifier
```

Place your `training.data` and `test.data` files in the same directory as `decisiontree.py`.

---

## Usage

Run the script directly:

```bash
python decisiontree.py
```

You’ll see printed F1 scores for each validation set, plus the best validation ranges for Gain Ratio and Gini Index.

---

## Dataset Format

The code expects:
- Last column = class label (`+` for positive, `-` for negative)
- Missing values marked as `?`
- Continuous and discrete attributes mixed is fine

Example row:
```
5.3,low,yes,+
```


## Project Status

This was originally a coursework project.  
Future improvements might include:
- Supporting pruning
- Handling multi-class classification
- Adding a small CLI for custom datasets

---

## License

This project is for educational purposes. Feel free to use or adapt with attribution.

---
