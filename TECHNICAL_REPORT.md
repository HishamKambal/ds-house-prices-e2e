

```md

\# Technical Report — House Prices Regression Pipeline



\## 1. Objective

The objective of this project is to predict house sale prices using the Ames Housing dataset.

The focus is on building a \*\*robust, leakage-safe, and reproducible machine learning pipeline\*\*

rather than maximizing leaderboard performance.



---



\## 2. Dataset

\- Source: Kaggle “House Prices: Advanced Regression Techniques”

\- Training data: `train.csv` (features + `SalePrice`)

\- Test data: `test.csv` (features only)



Raw datasets are stored locally and excluded from version control.



---



\## 3. Exploratory Data Analysis (EDA)

EDA revealed the following key insights:

\- `SalePrice` is heavily right-skewed.

\- A log transformation (`log1p`) significantly reduces skewness.

\- Numerous features contain missing values requiring imputation.

\- Several numeric variables show strong correlation with the target.

\- Some extreme outliers exist and influence linear models.



These findings directly informed preprocessing and modeling decisions.



---



\## 4. Target Transformation

\- Modeled target: `log1p(SalePrice)`

\- Inference output: transformed back using `expm1`



\*\*Rationale:\*\*  

Log transformation stabilizes variance, improves linear assumptions, and typically reduces RMSE.



---



\## 5. Preprocessing Strategy

All preprocessing is performed inside an sklearn `Pipeline` to prevent data leakage.



\### Numeric Features

\- Median imputation (`SimpleImputer(strategy="median")`)



\### Categorical Features

\- Most-frequent imputation

\- One-hot encoding with `handle\_unknown="ignore"`



Implementation uses `ColumnTransformer` to apply transformations selectively.



---



\## 6. Models Evaluated

The following regression models were evaluated:



\- Linear Regression

\- Ridge Regression (α = 10)

\- Lasso Regression (α = 0.0005)

\- Random Forest Regressor (400 trees)

\- Gradient Boosting Regressor



All models were wrapped in the same preprocessing pipeline for fair comparison.



---



\## 7. Evaluation Methodology

\- Cross-validation: 5-fold KFold

\- Shuffle enabled with fixed random seed (42)

\- Metric: RMSE on `log1p(SalePrice)`



RMSE was computed using:

neg\_root\_mean\_squared\_error



yaml

Copy code

and converted back to positive RMSE.



---



\## 8. Results

Cross-validated RMSE results:



| Model                | Mean RMSE | Std RMSE |

|---------------------|-----------|----------|

| Gradient Boosting   | 0.13405   | 0.02019  |

| Lasso               | 0.14157   | 0.04396  |

| Random Forest       | 0.14506   | 0.01925  |

| Linear Regression   | 0.15275   | 0.03508  |

| Ridge Regression    | 0.19657   | 0.04509  |



\*\*Selected Model:\*\* GradientBoostingRegressor  

Chosen due to lowest mean RMSE and stable variance across folds.



---



\## 9. Artifacts

Generated artifacts:

\- `outputs/model.joblib` — full preprocessing + model pipeline

\- `reports/metrics.json` — CV metrics and model comparison

\- `outputs/submission.csv` — Kaggle submission file



Artifacts are generated programmatically and excluded from Git.



---



\## 10. Limitations \& Future Improvements

\*\*Limitations\*\*

\- No hyperparameter tuning

\- Limited feature engineering

\- Baseline tree models only



\*\*Future Enhancements\*\*

\- Hyperparameter optimization

\- Feature engineering (polynomial, interactions)

\- Advanced gradient boosting libraries

\- Migration from notebooks to production scripts

\- CI testing and experiment tracking

