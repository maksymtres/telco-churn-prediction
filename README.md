# telco-churn-prediction
ML model to predict telecom customer churn with SHAP interpretability (ROC-AUC focus)

## Project overview
Telecom operator **TeleDom** wants to reduce customer churn. The business plan is to offer promo codes and special deals to subscribers who are likely to cancel their service.  
This project builds a **binary classification model** that predicts whether a customer will churn, using personal data, contract details, and information about connected services.

**Goal:** build a high-quality churn prediction model (primary metric: **ROC-AUC**) and interpret the drivers of churn to support retention actions.

## Tech stack
- **Python** (notebook kernel: 3.9.23)
- **pandas**, **numpy**
- **scikit-learn** (Pipeline, ColumnTransformer, models, metrics)
- **xgboost**
- **optuna** (hyperparameter tuning)
- **shap** (model interpretability)
- **matplotlib**, **seaborn**
- **phik** (correlation analysis)

## Dataset
The dataset is stored in 4 CSV files and merged by `customer_id`:
- `datasets/contract_new.csv` — contract, payments, dates
- `datasets/personal_new.csv` — personal features
- `datasets/internet_new.csv` — internet services
- `datasets/phone_new.csv` — telephony

### Target
Churn is derived from the contract end date:
- `churn = 1` if `end_date` is **present** (customer left)
- `churn = 0` if `end_date` is **missing** (customer is active)

### Feature engineering
- `tenure` is computed from dates as the customer lifetime:
  - from `begin_date` to `end_date` for churned customers,
  - from `begin_date` to a fixed snapshot date for active customers.

## Approach
1. Data loading and quality checks (types, missing values, duplicates)
2. Merging tables into a single dataset
3. Preprocessing with `ColumnTransformer`:
   - categorical: `SimpleImputer` + `OneHotEncoder`
   - numeric: `SimpleImputer` + `MinMaxScaler`
4. Model training and comparison:
   - DecisionTreeClassifier
   - LogisticRegression
   - RandomForestClassifier
   - GradientBoostingClassifier
   - XGBoost (XGBClassifier)
5. Hyperparameter tuning with **Optuna**
   - 5-fold **StratifiedKFold**
   - objective metric: **ROC-AUC**
   - random seed: `RANDOM_STATE = 171125`
6. Model interpretation with **SHAP**
7. Final evaluation (ROC curve, confusion matrix)

## Results
**Best model:** `GradientBoostingClassifier`

**Quality:**
- ROC-AUC (CV): **0.855**
- ROC-AUC (test): **0.929**
- Accuracy (test): **0.922**
- Baseline (DummyClassifier): ROC-AUC ≈ **0.50**

**Confusion matrix (test):**
- False Positives (FP): **25**
- False Negatives (FN): **102**
- True Positives (TP): **173**

**Interpretation (SHAP):**
Key churn drivers include:
- **tenure** (customer lifetime)
- **monthly charges**
- contract type (e.g., month-to-month vs long-term)
- connected services (support/security/backup features often reduce churn risk)

## How to run

### System requirements
- Python **3.9+**
- Jupyter Notebook / JupyterLab

### Installation
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn xgboost optuna shap phik matplotlib seaborn jupyter
```
