# Machine Learning From Scratch

Core ML algorithms implemented from scratch using only NumPy, no scikit learn, no high level APIs. The focus is on understanding the math, getting the implementation details right and evaluating properly.

## Algorithms

**Linear Regression** (`algorithms/linear_regression_gd.py`)
Batch gradient descent with multi feature support, L2 regularization, early stopping and convergence monitoring via loss curves.

**Logistic Regression** (`algorithms/logistic_regression_gd.py`)
Binary classification with sigmoid activation, cross entropy loss, feature scaling, numerical stability handling and train/test evaluation with full classification metrics.

**k-Nearest Neighbors** (`algorithms/knn_classifier.py`)
Distance based classification with majority voting, Euclidean distance and train/test evaluation.

## Evaluation & Utilities

`utils/metrics.py` provides accuracy, precision, recall, F1-score and confusion matrix computation, also implemented from scratch.

## Setup

```bash
pip install -r requirements.txt
python algorithms/linear_regression_gd.py
python algorithms/logistic_regression_gd.py
python algorithms/knn_classifier.py
```

## Built With

Python · NumPy · Matplotlib
