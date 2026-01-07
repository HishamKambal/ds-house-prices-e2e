# House Prices (Ames) — End-to-End Data Science Pipeline

This repository contains a complete, portfolio-grade data science workflow built on Kaggle’s **House Prices: Advanced Regression Techniques** dataset. It demonstrates an end-to-end approach a data scientist would use in practice: data inspection, exploratory analysis, leakage-safe preprocessing, cross-validated model comparison, objective model selection, and reproducible artifact generation.

## Project Goals
1. **Understand the data** through focused EDA (target distribution, missingness, feature signal, outliers).
2. **Build a robust preprocessing pipeline** that can handle mixed numeric/categorical data consistently.
3. **Train and compare multiple regression models** using cross-validation and an appropriate metric (RMSE).
4. **Select the best model** based on CV performance (mean RMSE) rather than a single split.
5. **Persist outputs** (metrics + best model + optional Kaggle submission) in a reproducible manner.

---

## Dataset
Source: Kaggle competition “House Prices: Advanced Regression Techniques”.

- Competition page:
  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Required files:
  - `train.csv` (features + target `SalePrice`)
  - `test.csv` (features only)

Place the files locally under:
data/raw/

yaml
Copy code

> Note: Raw datasets are intentionally excluded from Git to keep the repository lightweight and to follow best practices.

---

## Repository Structure
```text
ds-house-prices-e2e/
├── data/
│   ├── raw/                 # Kaggle CSV files (ignored by Git)
│   └── processed/           # optional (ignored by Git)
├── notebooks/
│   ├── 01_eda_house_prices.ipynb
│   └── 02_train_and_select_model.ipynb
├── outputs/                 # generated artifacts (typically ignored by Git)
├── reports/                 # metrics reports (often ignored; optionally commit metrics.json)
├── src/                     # optional: production scripts (future improvement)
├── docs/                    # optional: design notes (future improvement)
├── .gitignore
├── requirements.txt
├── README.md
└── TECHNICAL_REPORT.md
Method Overview (High-Level)
Target
Target variable: SalePrice

Transformation: use log1p(SalePrice) during training and evaluation; convert back with expm1 for final predictions.

Preprocessing (Leakage-Safe)
Implemented with sklearn using a single Pipeline:

Numeric features: median imputation

Categorical features: most-frequent imputation + one-hot encoding (handle_unknown="ignore")

Model Comparison
Candidate models are trained and evaluated under the same preprocessing pipeline and compared using:

5-fold KFold cross-validation (shuffle=True, random_state=42)

RMSE computed on the transformed target

Model Selection
Select the best model by:

lowest mean CV RMSE

How to Run (Windows / Anaconda)
1) Create and activate an environment
From the repository root:

bat
Copy code
cd /d C:\GitHub\portfolio\ds-house-prices-e2e
conda create -n houseprices python=3.10 -y
conda activate houseprices
2) Install dependencies
bat
Copy code
pip install -r requirements.txt
3) Ensure data exists
Confirm the files exist locally:

bat
Copy code
dir data\raw
You should see:

train.csv

test.csv

4) Run notebooks
Launch Jupyter from the project root (best practice):

bat
Copy code
jupyter notebook
Then run:

notebooks/01_eda_house_prices.ipynb

notebooks/02_train_and_select_model.ipynb

Outputs
When 02_train_and_select_model.ipynb completes, it generates:

outputs/model.joblib
Serialized best pipeline (preprocessing + estimator)

reports/metrics.json
CV summary, best model, and all candidate results

outputs/submission.csv (optional)
Kaggle submission file with Id and predicted SalePrice

These artifacts are generated automatically and are typically excluded from Git.

Results (Example from a Baseline Run)
The baseline workflow typically selects a gradient-boosted model (e.g., GradientBoostingRegressor) as a strong default. Your exact best model and RMSE are recorded in reports/metrics.json.

Quality & Best Practices
Reproducibility through fixed random seeds

Leakage-safe preprocessing inside the model pipeline

Cross-validated evaluation for reliable model comparison

Clean separation of raw data vs code vs generated artifacts

Next Improvements
Explicitly drop identifier columns (e.g., Id) from features

Add hyperparameter tuning (RandomizedSearchCV / Optuna)

Add stronger GBDT implementations (if allowed) and calibration checks

Move training logic from notebooks into src/ scripts

Add CI (linting + tests) and experiment tracking