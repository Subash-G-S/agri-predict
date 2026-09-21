# Agri-Predict: Machine Learning Capstone for Agricultural Intelligence

**Course:** 23CSE301 Machine Learning  
**Academic Year:** 2026-27  
**Degree:** B.Tech. Computer Science and Engineering (Year III)  
**Milestone:** Review 1 (Full Regression Track + Classification Track Part A)

---

## Team Members and Responsibilities

| Contributor | Email | Primary Responsibilities (Review 1) |
|---|---|---|
| **Subash Gopi** | `subashgopi2006@gmail.com` | Regression dataset audit & EDA foundation, feature relationship investigation, Linear, Ridge, Lasso, ElasticNet, and Polynomial Regression. |
| **Somesh K. Bharathwaj** | `somesh.k.bharathwaj@gmail.com` | Regression non-linear models (Decision Tree, Random Forest, Gradient Boosting, SVR, KNN), hyperparameter tuning, 5-fold CV, error diagnostics. |
| **Akhil Senthil** | `akhilsenthil696@gmail.com` | Classification dataset audit, multi-class EDA, preprocessing & stratified splitting, Part-A classifiers (Logistic, KNN, Naive Bayes, Decision Tree, SVC), confusion matrix analysis, 5-fold CV, documentation. |

---

## Project Overview

Agri-Predict applies machine learning techniques to Indian agricultural data to support evidence-based farming decisions. The project addresses two core analytical tracks for Review 1:

1. **Crop Yield Prediction (Regression Track):** Estimating crop yield (yield per unit area) using land area, input quantities (fertilizer, pesticide), weather conditions (rainfall, temperature profiles), and agro-climatic context (crop, season, state).
2. **Crop Recommendation (Classification Track Part A):** Recommending the optimal crop label from soil nutrient parameters (Nitrogen, Phosphorus, Potassium, pH) and micro-climate indicators (temperature, humidity, rainfall).

---

## Problem Statements

### Regression Track
Predict agricultural crop yield across diverse Indian states, seasons, and crop varieties using historical meteorological, input, and area records, and benchmark ten required regression algorithms on an identical evaluation split.

> **Important Feature Decision:** Exploratory analysis confirmed that recorded `Yield` is mathematically defined as `Production / Area` ($r = 0.9965$). Because total harvested production is an outcome only known *post-harvest*, including `Production` as an input would introduce circular data leakage. `Production` is formally excluded to formulate a genuine pre-harvest prediction model.

### Classification Track (Part A)
Classify 22 crop categories from 7 soil and climate measurements using the five Part-A baseline classifiers required for Review 1, establishing an interpretable multi-class benchmark.

---

## Datasets

Both raw datasets reside in the `data/` directory and are preserved without modification:

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

## Review 1 Algorithms

### Regression (10 Algorithms Evaluated)
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

### Classification Part A (5 Algorithms Evaluated)
1. Logistic Regression (Multi-class with `StandardScaler`)
2. K-Nearest Neighbors Classifier ($k = 5$)
3. Gaussian Naive Bayes (Probabilistic conditional independence)
4. Decision Tree Classifier (Gini impurity)
5. Support Vector Classifier (SVC with RBF kernel)

---

## Repository Structure

```
agri-predict/
├── README.md                          # Project documentation and results
├── requirements.txt                   # Environment dependencies
├── data/
│   ├── Crop Yeild Data.csv            # Raw regression dataset
│   └── Crop_recommendation.csv        # Raw classification dataset
└── notebooks/
    ├── regression.ipynb               # Full Review 1 regression pipeline (18 sections)
    └── classification.ipynb           # Full Review 1 classification pipeline (14 sections)
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
Both notebooks are pre-executed with all cell outputs, metric summaries, and plots visible.

---

## Review 1 Results Summary

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

#### Hyperparameter Tuning (Top 2 Models via GridSearchCV)
- **Gradient Boosting:** Tuned with `learning_rate=0.15`, `max_depth=3`, `n_estimators=150`  
  $\rightarrow$ $R^2$ improved from **0.9703** to **0.9709**, RMSE reduced to **152.70**.
- **Random Forest:** Tuned with `max_depth=15`, `n_estimators=150`  
  $\rightarrow$ Achieved constrained depth generalization with $R^2 = \mathbf{0.9605}$, RMSE = **178.00**.

#### 5-Fold Cross Validation (Training Set)
- **Tuned Gradient Boosting:** Mean $R^2 = \mathbf{0.9449} \pm 0.0613$
- **Tuned Random Forest:** Mean $R^2 = \mathbf{0.9449} \pm 0.0450$

---

### 2. Classification Track Part A Benchmark (Stratified 80/20 Split, $N = 440$)

| Rank | Model | Accuracy | Weighted F1 |
|:---:|---|:---:|:---:|
| 1 | **Gaussian Naive Bayes** | **99.55%** | **0.9954** |
| 2 | **Support Vector Classifier (SVC)** | **98.41%** | **0.9840** |
| 3 | Decision Tree Classifier | 97.95% | 0.9794 |
| 4 | K-Nearest Neighbors (KNN) | 97.95% | 0.9793 |
| 5 | Logistic Regression | 97.27% | 0.9725 |

#### 5-Fold Cross Validation (Training Set, Top 2 Classifiers)
- **Gaussian Naive Bayes:** Mean Weighted F1 = $\mathbf{0.9943} \pm 0.0051$
- **Support Vector Classifier (SVC):** Mean Weighted F1 = $\mathbf{0.9762} \pm 0.0074$

---

## AI Assistance Disclosure

In accordance with Section 7.5 of the 23CSE301 Capstone Guidelines, generative AI tools were utilized exclusively for code scaffolding, repetitive boilerplate assistance, and Markdown table formatting. All exploratory data analysis, data quality investigations, feature selection decisions (including the exclusion of `Production`), model interpretations, and conclusions were independently conducted and reasoned by the project team.
