\# House Prices (Ames) — End-to-End Data Science Project



This repository contains a \*\*full end-to-end data science workflow\*\* based on Kaggle’s  

\*\*House Prices: Advanced Regression Techniques\*\* competition.



The project demonstrates how a data scientist approaches:

\- Exploratory data analysis (EDA)

\- Data preprocessing

\- Leakage-safe pipelines

\- Cross-validated model comparison

\- Best model selection and artifact generation



---



\## Dataset

Source: Kaggle — House Prices: Advanced Regression Techniques  

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques



Raw data files (`train.csv`, `test.csv`) are stored locally under:

data/raw/



yaml

Copy code

and are \*\*not committed\*\* to GitHub.



---



\## Project Structure

```text

ds-house-prices-e2e/

│

├── data/

│   ├── raw/            # Kaggle CSV files (ignored by git)

│   └── processed/

│

├── notebooks/

│   ├── 01\_eda\_house\_prices.ipynb

│   └── 02\_train\_and\_select\_model.ipynb

│

├── outputs/            # Generated models \& predictions (ignored)

├── reports/            # Metrics \& evaluation reports (ignored)

│

├── README.md

├── TECHNICAL\_REPORT.md

└── .gitignore

Notebooks Overview

1️⃣ Exploratory Data Analysis

01\_eda\_house\_prices.ipynb



Target distribution and skewness analysis



Missing value inspection



Numeric \& categorical feature analysis



Correlation and outlier checks



Modeling decisions justified from data



2️⃣ Model Training \& Selection

02\_train\_and\_select\_model.ipynb



Leakage-safe preprocessing using Pipeline + ColumnTransformer



Multiple regression models trained



5-fold cross-validation using RMSE



Best model selected objectively



Artifacts saved for reuse



Modeling Summary

Target transformation: log1p(SalePrice)



Numeric features: median imputation



Categorical features: most-frequent imputation + one-hot encoding



Evaluation: 5-fold cross-validation (RMSE)



Best Model

GradientBoostingRegressor



CV RMSE (log target):





0.13405 ± 0.02019



Outputs (Generated Locally)

outputs/model.joblib — trained pipeline



reports/metrics.json — cross-validation results



outputs/submission.csv — Kaggle-ready predictions



These files are generated locally and not tracked in Git.



Notes

This project prioritizes clean structure, reproducibility, and evaluation discipline

over leaderboard optimization.



For technical details and decisions, see TECHNICAL\_REPORT.md

