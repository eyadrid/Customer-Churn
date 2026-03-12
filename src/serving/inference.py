import os
import glob
import pandas as pd
import mlflow
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


model_base_path = os.path.join(CURRENT_DIR, "model")
run_folders = glob.glob(os.path.join(model_base_path, "4e663*"))

if not run_folders:
    run_folders = [f.path for f in os.scandir(model_base_path) if f.is_dir()]

if not run_folders:
    raise Exception(f"Aucun dossier de modèle trouvé dans {model_base_path}")

RUN_DIR = run_folders[0]
ARTIFACTS_DIR = os.path.join(RUN_DIR, "artifacts")
MODEL_MLFLOW_DIR = os.path.join(ARTIFACTS_DIR, "model")

try:
    model = mlflow.pyfunc.load_model(MODEL_MLFLOW_DIR)
    print(f"Modèle chargé avec succès depuis : {MODEL_MLFLOW_DIR}")
except Exception as e:
    raise Exception(f"Erreur lors du chargement du modèle MLflow : {e}")


feature_file = os.path.join(ARTIFACTS_DIR, "feature_columns.txt")

try:
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
        print(f"{len(FEATURE_COLS)} colonnes de caractéristiques chargées.")
    else:
        raise FileNotFoundError(f"Fichier {feature_file} introuvable.")
except Exception as e:
    raise Exception(f"Erreur chargement feature_columns.txt : {e}")


BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df

def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        
        result = preds[0] if isinstance(preds, (list, pd.Series)) else preds
        
    except Exception as e:
        raise Exception(f"Échec de la prédiction : {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"