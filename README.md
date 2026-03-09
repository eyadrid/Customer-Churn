# Customer Churn Prediction

End-to-end machine learning project for telecom customer churn prediction, including:

- Data validation and preprocessing
- Feature engineering
- XGBoost model training with MLflow tracking
- FastAPI prediction service
- Integrated Gradio dashboard at `/ui`

## 1. Project Overview

This repository trains a binary classifier (`Churn` = yes/no) using the Telco Customer Churn dataset, logs runs with MLflow, and serves predictions through a FastAPI API and a Gradio web interface.

Main pipeline flow:

1. Load raw data
2. Validate quality with Great Expectations
3. Preprocess and engineer features
4. Train XGBoost model
5. Log metrics/artifacts to MLflow
6. Serve model for real-time inference

## 2. Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- Gradio
- XGBoost + scikit-learn
- MLflow
- Great Expectations
- Pandas / NumPy
- Optuna (for tuning scripts)
- Docker

## 3. Repository Structure

Key folders/files:

- `src/data/`: loading and preprocessing
- `src/features/`: feature engineering
- `src/models/`: training/evaluation/tuning utilities
- `src/serving/inference.py`: model loading + prediction transform
- `src/app/main.py`: FastAPI app + Gradio UI
- `scripts/run_pipeline.py`: full training pipeline
- `scripts/prepare_processed_data.py`: build processed dataset
- `scripts/test_*.py`: script-based checks and API tests
- `data/raw/`: source dataset
- `data/processed/`: processed training data
- `artifacts/`: feature list and preprocessing artifacts
- `mlruns/`: MLflow run artifacts

## 4. Setup

### 4.1 Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the Training Pipeline

From the project root:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn --threshold 0.35 --test_size 0.2 --experiment "Telco Churn"
```

Outputs created/updated:

- `data/processed/telco_churn_processed.csv`
- `artifacts/feature_columns.json`
- `artifacts/preprocessing.pkl`
- MLflow run data in `mlruns/`

## 6. Launch API and Dashboard

Start server:

```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Available endpoints:

- API root: `http://127.0.0.1:8000/`
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`
- Gradio dashboard: `http://127.0.0.1:8000/ui`

## 7. Prediction API Example

### 7.1 cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"gender": "Male",
		"Partner": "Yes",
		"Dependents": "No",
		"PhoneService": "Yes",
		"MultipleLines": "No",
		"InternetService": "Fiber optic",
		"OnlineSecurity": "No",
		"OnlineBackup": "Yes",
		"DeviceProtection": "No",
		"TechSupport": "No",
		"StreamingTV": "Yes",
		"StreamingMovies": "Yes",
		"Contract": "Month-to-month",
		"PaperlessBilling": "Yes",
		"PaymentMethod": "Electronic check",
		"tenure": 5,
		"MonthlyCharges": 70.35,
		"TotalCharges": 350.75
	}'
```

### 7.2 Python

```python
import requests

payload = {
		"gender": "Male",
		"Partner": "Yes",
		"Dependents": "No",
		"PhoneService": "Yes",
		"MultipleLines": "No",
		"InternetService": "Fiber optic",
		"OnlineSecurity": "No",
		"OnlineBackup": "Yes",
		"DeviceProtection": "No",
		"TechSupport": "No",
		"StreamingTV": "Yes",
		"StreamingMovies": "Yes",
		"Contract": "Month-to-month",
		"PaperlessBilling": "Yes",
		"PaymentMethod": "Electronic check",
		"tenure": 5,
		"MonthlyCharges": 70.35,
		"TotalCharges": 350.75,
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.status_code)
print(r.json())
```

## 8. MLflow Tracking

Pipeline logs:

- Params: model type, threshold, test split
- Metrics: precision, recall, F1, ROC AUC, train/predict time, data quality status
- Artifacts: model, feature columns, preprocessing metadata

To inspect runs:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## 9. Utility Scripts

- Prepare processed data:

```bash
python scripts/prepare_processed_data.py
```

- Data/features test flow:

```bash
python scripts/test_pipeline_data_features.py
```

- Modeling/tuning test flow:

```bash
python scripts/test_pipeline_modeling.py
```

- FastAPI request test (requires API running):

```bash
python scripts/test_fastapi.py
```

## 10. Docker

Build image:

```bash
docker build -t customer-churn -f dockerfile .
```

Run container:

```bash
docker run --rm -p 8000:8000 customer-churn
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/ui`

## 11. Notes and Troubleshooting

- If prediction returns model-not-loaded messages, run the training pipeline first.
- `src/serving/inference.py` first looks for artifacts in `src/serving/model/`, then falls back to latest `mlruns` model.
- Ensure your request payload uses the exact fields expected by `CustomerData` in `src/app/main.py`.