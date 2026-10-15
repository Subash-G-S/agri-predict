# Agri-Predict: Machine Learning Capstone for Agricultural Intelligence

**Course:** 23CSE301 Machine Learning  
**Academic Year:** 2026-27  
**Degree:** B.Tech. Computer Science and Engineering (Year III)  
**Milestone:** Review 1 & Review 2 (Full Regression Track + Full Classification Track Parts A & B)

---

## Team Members and Responsibilities

| Contributor | Email | Primary Responsibilities (Review 1 & Review 2) |
|---|---|---|
| **Subash Gopi** | `subashgobi2006@gmail.com` | **Review 1:** Regression dataset audit & EDA foundation, Linear/Ridge/Lasso/ElasticNet/Poly. <br>**Review 2:** AdaBoost classifier implementation & depth tuning, Multi-Layer Perceptron (MLP) neural network pipeline with scaling, documentation & integration. |
| **Somesh K. Bharathwaj** | `somesh.k.bharathwaj@gmail.com` | **Review 1:** Regression non-linear models (Decision Tree, RF, GB, SVR, KNN), tuning, 5-fold CV. <br>**Review 2:** Gradient Boosting classifier implementation, Review 1 baseline freeze & integrity checks, consolidated 10-classifier comparative benchmark. |
| **Akhil Senthil** | `akhilsenthil696@gmail.com` | **Review 1:** Classification dataset audit, multi-class EDA, Part A classifiers (Logistic, KNN, GNB, DT, SVC). <br>**Review 2:** Random Forest classifier & feature importance, Decision Tree Bagging ensemble, 5-fold cross-validated hyperparameter tuning on champion model. |

---

## Project Overview

Agri-Predict applies machine learning techniques to Indian agricultural data to support evidence-based farming decisions. The project addresses two core analytical tracks:

1. **Crop Yield Prediction (Regression Track):** Estimating crop yield (yield per unit area) using land area, input quantities (fertilizer, pesticide), weather conditions (rainfall, temperature profiles), and agro-climatic context (crop, season, state).
2. **Crop Recommendation (Classification Track):**
   - **Part A (Review 1):** Five foundational classifiers (Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC).
   - **Part B (Review 2):** Five advanced ensemble & neural classifiers (Random Forest, AdaBoost, Gradient Boosting, Bagging with Decision Tree, MLP Neural Network), followed by a consolidated 10-model benchmark and 5-fold cross-validated hyperparameter tuning.

---

## Problem Statements

### Regression Track
Predict agricultural crop yield across diverse Indian states, seasons, and crop varieties using historical meteorological, input, and area records, and benchmark ten required regression algorithms on an identical evaluation split.

> **Important Feature Decision:** Exploratory analysis confirmed that recorded `Yield` is mathematically defined as `Production / Area` ($r = 0.9965$). Because total harvested production is an outcome only known *post-harvest*, including `Production` as an input would introduce circular data leakage. `Production` is formally excluded to formulate a genuine pre-harvest prediction model.

### Classification Track (Parts A & B)
Classify 22 crop categories from 7 soil and climate measurements (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`). All ten required algorithms are evaluated on the exact same 80:20 stratified split ($N_{\text{train}}=1,760$, $N_{\text{test}}=440$, `random_state=42`) using five mandatory evaluation metrics: Accuracy, Precision, Recall, Weighted F1, and Multiclass OvR ROC-AUC.

---

## Datasets

Both raw datasets reside in the `data/` directory and are strictly preserved without modification:

1. **`data/Crop Yeild Data.csv`**
   - **Rows:** 19,689 records | **Columns:** 13
   - **Target:** `Yield` (continuous, metric tons or units per hectare)
   - **Features:** `Crop`, `Crop_Year`, `Season`, `State`, `Area`, `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, `Avg_Temperature`, `Max_Temperature`, `Min_Temperature`
   - **Engineered Feature:** `Temperature_Range = Max_Temperature - Min_Temperature`

2. **`data/Crop_recommendation.csv`**
   - **Rows:** 2,200 records | **Columns:** 8
   - **Target:** `label` (22 crop classes, exactly 100 observations each)
   - **Features:** `N` (Nitrogen), `P` (Phosphorus), `K` (Potassium), `temperature`, `humidity`, `ph`, `rainfall`

---

## Evaluated Algorithms

### Regression (10 Algorithms Evaluated — Review 1)
1. Linear Regression (Baseline)
2. Ridge Regression ($L_2$ regularization)
3. Lasso Regression ($L_1$ regularization)
4. ElasticNet Regression (Combined $L_1 + L_2$)
5. Polynomial Regression (Degree = 2 on continuous features)
6. Decision Tree Regressor
7. Random Forest Regressor (Ensemble bagging)
8. Gradient Boosting Regressor (Ensemble boosting)
9. Support Vector Regressor (SVR with RBF kernel)
10. K-Nearest Neighbors Regressor (Distance-based)

### Classification Part A (5 Foundational Classifiers — Review 1)
1. Logistic Regression (Multi-class with `StandardScaler`)
2. K-Nearest Neighbors Classifier ($k = 5$)
3. Gaussian Naive Bayes (Probabilistic conditional independence)
4. Decision Tree Classifier (Gini impurity)
5. Support Vector Classifier (SVC with RBF kernel)

### Classification Part B (5 Advanced Classifiers — Review 2)
6. **Random Forest Classifier** (Decorrelated tree bagging with Gini feature importance)
7. **AdaBoost Classifier** (Adaptive boosting with shallow decision tree base estimator, `max_depth=3`)
8. **Gradient Boosting Classifier** (Stage-wise additive expansion minimizing multi-class deviance loss)
9. **Bagging Classifier** (Bootstrap aggregating with unpruned `DecisionTreeClassifier` base estimator)
10. **Multi-Layer Perceptron (MLP) Classifier** (Feedforward artificial neural network with training-only `StandardScaler`)

---

## Repository Structure

```
agri-predict/
├── README.md                          # Project documentation and results
├── requirements.txt                   # Environment dependencies
├── data/
│   ├── Crop Yeild Data.csv            # Raw regression dataset (preserved)
│   └── Crop_recommendation.csv        # Raw classification dataset (preserved)
└── notebooks/
    ├── regression.ipynb               # Full Review 1 regression pipeline (18 sections)
    └── classification.ipynb           # Full Review 1 & Review 2 classification pipeline (25 sections)
```

---

## Environment Setup and Execution

### 1. Clone Repository & Create Environment
```bash
git clone https://github.com/Subash-G-S/agri-predict.git
cd agri-predict

# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook notebooks/regression.ipynb
jupyter notebook notebooks/classification.ipynb
```
Both notebooks are pre-executed from top to bottom with all cell outputs, metric summaries, confusion matrices, and plots fully visible.

---

## Experimental Results Summary

### 1. Regression Track Benchmark (80/20 Test Split, $N = 3,938$)

| Rank | Model | $R^2$ Score | RMSE | MAE |
|:---:|---|:---:|:---:|:---:|
| 1 | **Gradient Boosting Regressor** | **0.9703** | **154.32** | 13.11 |
| 2 | **Random Forest Regressor** | **0.9630** | **172.15** | **11.95** |
| 3 | Decision Tree Regressor | 0.9318 | 233.79 | 15.97 |
| 4 | KNN Regressor | 0.9182 | 256.00 | 19.02 |
| 5 | Linear Regression | 0.8105 | 389.68 | 63.91 |
| 6 | Polynomial Regression ($d=2$) | 0.8105 | 389.63 | 69.30 |
| 7 | Ridge Regression ($\alpha=1.0$) | 0.8103 | 389.88 | 64.21 |
| 8 | Lasso Regression ($\alpha=1.0$) | 0.8086 | 391.60 | 54.27 |
| 9 | ElasticNet ($\alpha=0.1, l_1=0.5$) | 0.4447 | 594.39 | 94.46 |
| 10 | Support Vector Regressor (SVR) | 0.0039 | 893.39 | 73.19 |

---

### 2. Consolidated 10-Algorithm Classification Benchmark (Stratified 80/20 Split, $N = 440$)

All 10 classifiers are evaluated on the exact same held-out test observations. Averaging convention for Precision, Recall, and F1 is **weighted** across all 22 classes. ROC-AUC is evaluated via **One-vs-Rest (OvR)** with weighted averaging on true predicted probability distributions.

| Rank | Model | Track | Accuracy | Precision | Recall | Weighted F1 | OvR ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Random Forest Classifier** | Part B | **99.55%** | **0.9957** | **0.9955** | **0.9955** | **1.0000** |
| 2 | **AdaBoost Classifier** | Part B | **99.55%** | **0.9957** | **0.9955** | **0.9955** | 0.9998 |
| 3 | **Gaussian Naive Bayes** | Part A | 99.55% | 0.9956 | 0.9955 | 0.9954 | 1.0000 |
| 4 | **Bagging Classifier (Decision Tree)** | Part B | 99.32% | 0.9936 | 0.9932 | 0.9932 | 0.9999 |
| 5 | **MLP Classifier (Neural Network)** | Part B | 99.09% | 0.9915 | 0.9909 | 0.9909 | 1.0000 |
| 6 | **Gradient Boosting Classifier** | Part B | 98.86% | 0.9892 | 0.9886 | 0.9887 | 1.0000 |
| 7 | **Support Vector Classifier (SVC)** | Part A | 98.41% | 0.9845 | 0.9841 | 0.9840 | 1.0000 |
| 8 | **Decision Tree Classifier** | Part A | 97.95% | 0.9806 | 0.9795 | 0.9794 | 0.9893 |
| 9 | **K-Nearest Neighbors (KNN)** | Part A | 97.95% | 0.9806 | 0.9795 | 0.9793 | 0.9987 |
| 10 | **Logistic Regression** | Part A | 97.27% | 0.9739 | 0.9727 | 0.9725 | 0.9998 |

---

### 3. Final Model Selection & Hyperparameter Tuning (Review 2 Rubric A3)

We selected the champion **Random Forest Classifier** for systematic hyperparameter tuning using **Stratified 5-Fold Cross-Validation** strictly on the training partition ($N_{\text{train}}=1,760$):
- **Search Method:** `GridSearchCV(cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_weighted')`
- **Search Space:** `n_estimators`: [100, 150, 200], `max_depth`: [None, 12, 16], `min_samples_leaf`: [1, 2]
- **Optimal Hyperparameters:** `{'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 100}`

#### Quantitative Comparison Before vs. After Tuning

| Evaluation Metric | Baseline Random Forest (`depth=None`) | Tuned Random Forest (`depth=12`, `leaf=1`) | Observed Metric Change |
|---|:---:|:---:|---|
| **Test Accuracy** | 99.55% | 99.55% | 0.0000 (Maintained peak 99.55%) |
| **Test Precision (Weighted)** | 0.9957 | 0.9957 | 0.0000 (Maintained peak precision) |
| **Test Recall (Weighted)** | 0.9955 | 0.9955 | 0.0000 (Maintained peak recall) |
| **Test Weighted F1** | 0.9955 | 0.9955 | 0.0000 (Maintained peak F1) |
| **Test OvR ROC-AUC** | 0.999995 | 1.000000 | **+0.000005** (Pushes to 1.000000) |
| **Stratified 5-Fold CV F1** | 0.9937 | 0.9949 | **+0.0012** (Generalization gain) |

**Key Agronomic & Technical Finding:**  
Restricting tree depth from unbounded growth (`depth=None`) to `depth=12` prevents individual decision paths from memorizing noisy micro-variations while preserving 99.55% test accuracy. Crucially, the cross-validated F1 score improves by **+0.0012**, proving enhanced model generalization and reducing inference memory footprint for precision agriculture deployments.

---

## AI Assistance Disclosure

In accordance with Section 7.5 of the 23CSE301 Capstone Guidelines, generative AI tools were utilized exclusively for code scaffolding, repetitive boilerplate assistance, and Markdown table formatting. All exploratory data analysis, data quality investigations, feature selection decisions (including the exclusion of `Production`), model interpretations, hyperparameter tuning design, and conclusions were independently conducted and reasoned by the project team.
