# Runtime and Duration Class Prediction in HPC Systems

This repository contains the code and data used for a series of experiments on runtime and duration class prediction in High-Performance Computing (HPC) systems.  
The goal of this project is to replace inaccurate user-provided runtime estimates with machine learning–based predictions to improve job scheduling and prioritization efficiency.

---

## Overview

This work represents the first phase of a larger project aimed at integrating ML techniques into HPC workload dispatching.  
It focuses on predicting the runtime (regression task) and duration class (classification task) of submitted jobs on the Marconi100 supercomputer (CINECA).

Machine learning methods are compared to traditional baselines:
- User-provided estimates  
- A simple historical heuristic  

and include:
- Decision Tree Regression
- Normalized Polynomial Regression (custom Ridge-based model)
- k-Nearest Neighbors (k-NN) Classification with 4 and 7 duration classes.

---

## Key Components

### **DataLoader**
Defined in `modules/data_loader.py`, this class:
- Loads the original PM100 dataset.
- Enriches it with additional job, historical, and system-level features.
- Defines and assigns duration classes:
  - 4-class version: _Very-Short, Short, Medium, Long_
  - 7-class version: _Very-Short, Short, Medium-Short, Medium, Medium-Long, Long, Very-Long_
- Splits data chronologically (70% train / 30% test).
- Saves resulting parquet files for downstream experiments.

### **RidgePolynomialRegressor (Normalized Polynomial)**
Defined in `modules/prediction_models.py`:

```python
class RidgePolynomialRegressor:
    """
    Offline polynomial regression model with ridge regularization.
    Performs polynomial feature expansion, standardization, and regression.
    """
    def __init__(self, degree=2, alpha=0.01):
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        partial = self.model.predict(X_scaled)
        return np.where(partial < 1, 1, np.where(partial > 86400, 86400, partial)).astype(int)

```

### Utilities

`modules/utils.py` includes functions for:

- Computing and displaying evaluation metrics.
- Plotting histograms and scatter plots for comparing predicted vs. actual runtimes.

---

### Experimental Setup

Two feature sets are used:

- SET 1: minimal, directly available at job submission.
- SET 2: extended, combining historical, job-specific, and system-state features.

Experiments are run using the provided notebooks to compare:

- Baseline methods (user estimate, heuristic)
- Machine learning regressors/classifiers

---

### Results Summary


| Task                           | Best Model            | Feature Set | Key Result                                             |
| ------------------------------ | --------------------- | ----------- | ------------------------------------------------------ |
| **Regression**                 | Normalized Polynomial | SET 2       | Lowest average error (~25% improvement over heuristic) |
| **Classification (4 classes)** | k-NN                  | SET 2       | Accuracy ≈ 0.80                                        |
| **Classification (7 classes)** | k-NN                  | SET 2       | Accuracy ≈ 0.71                                        |


Machine learning methods, especially the Normalized Polynomial and k-NN (SET 2), outperform both baselines, showing clear benefits from feature enrichment.

More details are available in the accompanying PDF report.
