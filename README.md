# 🫀 Heart Disease Prediction using Machine Learning

In this project, the performance of an SVM is compared with a Random Forest Classifier using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The data is fetched from the UC IRvine repo, pre processed to drop na values, and the clas imbalance is adjusted by grouping target values 1-4 together. This results in a binary target 0 for no risk and 1 for risk of heart disease. It is demonstrated that RandomForest performs better than SVM when the class imbalance is adjusted.

## 📊 Problem Overview

Heart disease remains one of the leading causes of death globally. Early prediction of heart conditions can help in timely medical intervention. This project uses patient attributes (like cholesterol level, resting ECG, etc.) to predict whether the patient is at risk of heart disease.

---

## Dataset

We use the **UCI Heart Disease Dataset**, fetched via the `ucimlrepo` Python package.

- **Original Targets**: 0 (no disease) to 4 (increasing severity)
- **Modified Targets**: For simplicity and clearer prediction:
  - `0` → No disease
  - `1-4` → Disease
  - This turns it into a **binary classification** task.

---

## Models Used

Two models were trained and compared:

| Model          | Training Accuracy | Validation Accuracy |
|----------------|-------------------|---------------------|
| SVM (Support Vector Machine) | 1.000               | 0.617               |
| Random Forest  | 1.000               | **0.883** (better metrics)   |

> 🔍**Random Forest performed better** on precision and recall for positive (disease) classes and achieiving higher accuracy across the board.

---

