import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import optuna
import os

print("Phase 2: Modeling with XGBoost")

# 1. Load Data
file_path = r"C:\Users\dridi\Desktop\Customer Churn\data\processed\telco_churn_processed.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find processed data at {file_path}. Run the main pipeline first.")

df = pd.read_csv(file_path)

# 2. Fix Target Column (Churn)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

# Remove any rows where Churn might be NaN after mapping
df = df.dropna(subset=["Churn"])
assert set(df["Churn"].unique()) <= {0, 1}, "Churn column contains invalid values."

# 3. Feature Engineering: Convert Categorical Strings to Numbers
print("Encoding categorical features...")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert objects to dummy variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Important: XGBoost requires bools (True/False) to be ints (1/0)
for col in X.select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)

print(f"Feature count after encoding: {X.shape[1]}")

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

# 5. Define Optuna Objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        # Handle class imbalance automatically
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Get probabilities for the positive class (Churn=1)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    y_pred = (proba >= THRESHOLD).astype(int)
    
    # We optimize for Recall (finding as many churners as possible)
    return recall_score(y_test, y_pred, pos_label=1)

# 6. Run Optimization
study = optuna.create_study(direction="maximize")
print("Starting hyperparameter optimization...")
study.optimize(objective, n_trials=30)

print("\n" + "="*30)
print("Optimization Finished!")
print(f"Best Recall Score: {study.best_value:.4f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*30)