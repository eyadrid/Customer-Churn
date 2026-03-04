import os
import pandas as pd
import mlflow
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

PROD_MODEL_PATH = os.path.join(CURRENT_DIR, "model")
PROD_FEATURE_FILE = os.path.join(CURRENT_DIR, "feature_columns.txt")


model = None
FEATURE_COLS = []

def load_production_artifacts():
    """Charge le modèle et la liste des colonnes depuis le dossier serving."""
    global model, FEATURE_COLS
    
    if os.path.exists(PROD_MODEL_PATH):
        try:
            model = mlflow.pyfunc.load_model(PROD_MODEL_PATH)
            print(f"Modèle chargé depuis : {PROD_MODEL_PATH}")
        except Exception as e:
            print(f"Erreur chargement modèle prod : {e}")
    
    if os.path.exists(PROD_FEATURE_FILE):
        try:
            with open(PROD_FEATURE_FILE, "r") as f:
                FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
            print(f"{len(FEATURE_COLS)} colonnes chargées depuis : {PROD_FEATURE_FILE}")
        except Exception as e:
            print(f"Erreur chargement colonnes prod : {e}")

def fallback_load_from_mlruns():
    """Si les fichiers ne sont pas en prod, cherche dans mlruns (Développement)."""
    global model, FEATURE_COLS
    if model is not None and FEATURE_COLS:
        return

    print("Recherche de secours dans mlruns...")
    search_pattern = os.path.join(BASE_DIR, "mlruns", "*", "*", "artifacts", "model")
    local_model_paths = glob.glob(search_pattern)

    if local_model_paths:
        latest_model = max(local_model_paths, key=os.path.getmtime)
        model = mlflow.pyfunc.load_model(latest_model)
        
        artifacts_dir = os.path.dirname(latest_model)
        feature_file = os.path.join(artifacts_dir, "feature_columns.txt")
        
        if os.path.exists(feature_file):
            with open(feature_file, "r") as f:
                FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
            print(f"Fallback : Modèle et colonnes chargés depuis mlruns.")
        else:
            print("Modèle trouvé mais feature_columns.txt manquant dans mlruns.")
    else:
        print("Aucun modèle trouvé. Veuillez lancer l'entraînement (run_pipeline.py).")

load_production_artifacts()
fallback_load_from_mlruns()


BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les données brutes pour correspondre exactement au format d'entraînement."""
    df = df.copy()
    
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
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)
    
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df


def predict(input_dict: dict) -> str:
    """Fonction appelée par l'API et Gradio."""
    if model is None or not FEATURE_COLS:
        return "Erreur : Le modèle n'est pas chargé."

    try:
        df_raw = pd.DataFrame([input_dict])
        df_transformed = _serve_transform(df_raw)
        
        preds = model.predict(df_transformed)
        
        if hasattr(preds, "tolist"):
            result = preds.tolist()[0]
        else:
            result = preds[0]
            
        return "Likely to churn" if result == 1 else "Not likely to churn"
        
    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"