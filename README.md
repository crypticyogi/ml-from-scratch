# Machine Learning From Scratch

## Overview
Implementations of core machine learning algorithms **from scratch** (no high-level model APIs),
focusing on clear code, learning dynamics, and correct evaluation.

## Implemented
### 1) Linear Regression (Gradient Descent)
- File: `linear_regression_gd.py`
- Demonstrates: MSE loss, batch gradient descent, convergence via loss curve

### 2) Logistic Regression (Gradient Descent)
- File: `logistic_regression_gd.py`
- Demonstrates: sigmoid, binary cross-entropy, decision surface, train/test evaluation + metrics

### 3) k-Nearest Neighbors (KNN)
- File: `knn_classifier.py`
- Demonstrates: distance-based classification, majority vote, simple train/test evaluation

## Utilities
- `metrics.py` — basic classification metrics (accuracy/precision/recall/F1, confusion counts)

## Tech Stack
Python, NumPy, Matplotlib

## How to run
```bash
pip install -r requirements.txt

python linear_regression_gd.py
python logistic_regression_gd.py
python knn_classifier.py
