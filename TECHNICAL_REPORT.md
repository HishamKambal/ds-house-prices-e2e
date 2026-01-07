```md
# Technical Report — House Prices (Ames) Regression Pipeline

## 1. Objective
This project predicts house sale prices (`SalePrice`) using Kaggle’s Ames Housing dataset. The repository is designed to showcase a rigorous, end-to-end machine learning workflow with clean engineering discipline: leakage-safe preprocessing, cross-validated evaluation, model comparison, and reproducible outputs.

## 2. Data Description
### 2.1 Source
Kaggle competition: “House Prices: Advanced Regression Techniques”.

### 2.2 Files
- `train.csv`: 1460 rows × 81 columns (features + target `SalePrice`)
- `test.csv`: 1459 rows × 80 columns (features only)

### 2.3 Target
- `SalePrice` is continuous and typically right-skewed.

## 3. Exploratory Data Analysis (EDA)
The EDA notebook (`01_eda_house_prices.ipynb`) focuses on:
- **Target distribution**: identifies skewness and motivates a log transform.
- **Missingness analysis**: ranks features by missing percent to plan imputation.
- **Feature signal inspection**:
  - correlation review for numeric predictors
  - boxplots/relationships for categorical predictors
- **Outlier inspection**: checks for extreme points that can influence regression.

### Key EDA conclusions
1. `SalePrice` is right-skewed → log-transform recommended.
2. Mixed numeric and categorical data requires structured preprocessing.
3. Missing values are substantial in multiple columns → imputation is required.
4. Strong relationships exist between `SalePrice` and common predictors (quality/area/basement-related features).

## 4. Target Transformation
### 4.1 Choice
We model:
- `y = log1p(SalePrice)`

### 4.2 Rationale
- reduces skew
- stabilizes variance
- improves linear model assumptions
- typically improves RMSE in price prediction tasks

### 4.3 Inference
Predictions are converted back via:
- `SalePrice_hat = expm1(y_hat)`

## 5. Preprocessing and Feature Handling
Preprocessing is implemented using `sklearn` Pipelines to ensure transformations are learned strictly within training folds and applied consistently at inference.

### 5.1 Column split
- Numeric columns: `select_dtypes(include=[number])`
- Categorical columns: `select_dtypes(exclude=[number])`

### 5.2 Numeric pipeline
- Median imputation:
  - `SimpleImputer(strategy="median")`
Rationale: robust to outliers and suitable for skewed numeric distributions.

### 5.3 Categorical pipeline
- Most-frequent imputation:
  - `SimpleImputer(strategy="most_frequent")`
- One-hot encoding:
  - `OneHotEncoder(handle_unknown="ignore")`
Rationale: consistent handling of missing categories; safe encoding when unseen categories appear.

### 5.4 Integration via ColumnTransformer
A `ColumnTransformer` applies numeric and categorical pipelines in parallel. The result is a single feature matrix that can be consumed by any downstream estimator.

### 5.5 Leakage prevention
All transformations are fit inside the cross-validation process via a top-level Pipeline:
- `Pipeline([("preprocess", ColumnTransformer(...)), ("model", estimator)])`

This prevents:
- target leakage
- validation fold contamination
- training/serving skew

## 6. Model Candidates
The comparison notebook evaluates multiple algorithm families under identical preprocessing:

1. **LinearRegression**
   - baseline model
   - establishes a performance floor
2. **Ridge Regression**
   - L2 regularization, helps multicollinearity after one-hot encoding
3. **Lasso Regression**
   - L1 regularization, can perform implicit feature selection
4. **RandomForestRegressor**
   - non-linear model, handles interactions, robust baseline for tabular data
5. **GradientBoostingRegressor**
   - strong general-purpose model for structured/tabular data

## 7. Evaluation Protocol
### 7.1 Cross-validation design
- 5-fold KFold:
  - `KFold(n_splits=5, shuffle=True, random_state=42)`
- Metric: RMSE on log target
  - `neg_root_mean_squared_error` (converted back to positive RMSE)

### 7.2 Selection criterion
Best model chosen by:
- minimum mean CV RMSE

### 7.3 Reporting
Results are saved to:
- `reports/metrics.json`

This report includes:
- best model name
- mean and std RMSE
- all candidate model results
- CV configuration
- target transform metadata

## 8. Results
The model comparison produces a ranked table of CV RMSE values. The selected best model is the one with lowest mean RMSE. The full trace of results is recorded in `reports/metrics.json`.

## 9. Artifacts and Reproducibility
Generated artifacts:
- `outputs/model.joblib`
  - serialized end-to-end Pipeline (preprocess + best estimator)
- `reports/metrics.json`
  - evaluation summary and model comparison
- `outputs/submission.csv` (optional)
  - Kaggle submission format

Reproducibility controls:
- fixed random seed (CV and relevant models)
- consistent pipeline transformations
- deterministic report generation

## 10. Limitations
- No extensive hyperparameter tuning
- No advanced feature engineering beyond standard preprocessing
- Baseline estimators only (competitive methods such as XGBoost/LightGBM are not included)
- Outlier handling is minimal (EDA-based review only)

## 11. Recommended Upgrades
If extending this project:
1. Explicitly drop identifier columns (e.g., `Id`) from model features.
2. Add feature engineering:
   - age features (YrSold - YearBuilt)
   - interaction terms and ratios (if beneficial)
   - ordinal encoding for known ordinal categoricals
3. Hyperparameter tuning with RandomizedSearchCV / Optuna.
4. Add robust model families (HistGradientBoosting, XGBoost/LightGBM if permitted).
5. Move training logic from notebooks into scripts (`src/train.py`) and add CI:
   - linting (ruff/flake8), formatting (black), and basic unit tests.