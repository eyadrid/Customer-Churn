# 📡 Telco Customer Churn Prediction

## Overview

Customer churn is a major business challenge for telecom companies because losing existing customers directly affects recurring revenue.
This project builds an end-to-end machine learning pipeline that predicts whether a telecom customer is likely to leave the service.

The workflow covers data loading, preprocessing, feature engineering, model training with XGBoost, experiment tracking with MLflow, and deployment through FastAPI with a Gradio user interface.

## Features

- Customer churn prediction using Machine Learning
- Model trained using XGBoost
- REST API built with FastAPI
- Interactive web interface using Gradio
- Clean and modular project structure

## Dataset

The project uses the IBM Telco Customer Churn dataset stored in the raw data folder.

The dataset includes customer information such as:

- Demographics
- Phone and internet services
- Contract details
- Billing and payment information
- Customer tenure and charges

Target variable:

- **Churn** (`Yes` / `No`)

Source: IBM Telco Customer Churn Dataset

## Model

The model used in this project is **XGBoost Classifier**.

Why XGBoost:

- Performs well on structured tabular data
- Handles non-linear relationships effectively
- Works well with engineered categorical and numeric features
- Delivers strong predictive performance for churn classification tasks

### Model Performance

Saved serving metrics from the current trained model:

- Precision: **49.04%**
- Recall: **82.09%**
- F1-score: **61.40%**
- ROC AUC: **83.67%**

### Why Recall Matters

In churn prediction, recall is one of the most important metrics because the business cost is asymmetric.

- **False Negative (FN):** the model predicts that a customer will stay, but the customer actually churns. This means the company misses the opportunity to intervene with a retention action.
- **False Positive (FP):** the model predicts that a customer will churn, but the customer would have stayed. This may lead to some unnecessary retention cost, but the customer is not lost.

In practice, missing a real churner is usually more expensive than targeting a loyal customer by mistake. For that reason, this project gives strong importance to **recall**, while also tracking **F1-score** to keep a balance between finding churners and limiting too many false alarms.

## Installation

Clone the repository:

```bash
git clone https://github.com/eyadrid/Customer-Churn.git
cd customer-churn
```

Create and activate a virtual environment if needed, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

### Start the API and Gradio app

Run the main application:

```bash
python src/app/main.py
```

The services will be available at:

- API root: `http://127.0.0.1:8000`
- Prediction endpoint: `http://127.0.0.1:8000/predict`
- Gradio interface: `http://127.0.0.1:8000/ui`

### Run the training pipeline

To retrain the model on the raw dataset:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv
```



