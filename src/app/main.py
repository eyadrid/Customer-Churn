from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.serving.inference import predict
except ImportError as e:
    print(f"Error: Could not import predict from src.serving.inference. {e}")
    sys.exit(1)


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting telecom customer churn",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}



class CustomerData(BaseModel):

    gender: str
    Partner: str
    Dependents: str

    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    tenure: int
    MonthlyCharges: float
    TotalCharges: float



@app.post("/predict")
def get_prediction(data: CustomerData):

    try:
        result = predict(data.dict())
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}



def gradio_interface(
    gender, Partner, Dependents,
    PhoneService, MultipleLines,
    InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV,
    StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod,
    tenure, MonthlyCharges, TotalCharges
):

    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        result = predict(data)

        if result == "Not likely to churn":
            return f"Customer Likely to Stay\n\nPrediction: {result}"

        elif result == "Likely to churn":
            return f"High Churn Risk\n\nPrediction: {result}"

        else:
            return f"Prediction: {result}"

    except Exception as e:
        return f"Error: {str(e)}"



css = """
.gradio-container {
    max-width: 1200px !important;
}

.prediction-box textarea {
    font-size:18px !important;
    font-weight:bold;
}

/* Predict button gradient */
#predict-btn {
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
    color: white !important;
    border: none !important;
    font-weight: bold;
    font-size: 18px;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.2);
}

#predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}
"""



with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown(
    """
    <div style="
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    ">

    <h1 style="margin-bottom:10px;">📡 Telco Customer Churn Predictor</h1>

    <p style="font-size:18px;">
    Predict whether a telecom customer is <b>likely to churn</b> using a trained
    <b>Machine Learning model (XGBoost)</b>.
    </p>

    </div>
    """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("## 👤 Demographics")

            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")

            gr.Markdown("## 📞 Phone Services")

            PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")

            MultipleLines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple Lines",
                value="No"
            )

        with gr.Column():

            gr.Markdown("## 🌐 Internet Services")

            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic"
            )

            OnlineSecurity = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Security",
                value="No"
            )

            OnlineBackup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Backup",
                value="No"
            )

            DeviceProtection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Device Protection",
                value="No"
            )

            TechSupport = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Tech Support",
                value="No"
            )

            StreamingTV = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming TV",
                value="Yes"
            )

            StreamingMovies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming Movies",
                value="Yes"
            )

    gr.Markdown("## 💳 Billing Information")

    with gr.Row():

        Contract = gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month"
        )

        PaperlessBilling = gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing",
            value="Yes"
        )

        PaymentMethod = gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method",
            value="Electronic check"
        )

    gr.Markdown("## 📊 Account Metrics")

    with gr.Row():

        tenure = gr.Number(
            label="Tenure (months)",
            value=1,
            minimum=0,
            maximum=100
        )

        MonthlyCharges = gr.Number(
            label="Monthly Charges ($)",
            value=85.0
        )

        TotalCharges = gr.Number(
            label="Total Charges ($)",
            value=85.0
        )

    with gr.Row():

        predict_btn = gr.Button("🔮 Predict Churn", elem_id="predict-btn")
        clear_btn = gr.Button("Clear")

    output = gr.Textbox(
        label="Prediction Result",
        lines=3,
        elem_classes="prediction-box"
    )

    predict_btn.click(
        gradio_interface,
        inputs=[
            gender, Partner, Dependents,
            PhoneService, MultipleLines,
            InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV,
            StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod,
            tenure, MonthlyCharges, TotalCharges
        ],
        outputs=output
    )

    clear_btn.click(lambda: "", None, output)



app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)



if __name__ == "__main__":

    import uvicorn

    print("→ API :       http://127.0.0.1:8000")
    print("→ Interface : http://127.0.0.1:8000/ui")

    uvicorn.run(app, host="0.0.0.0", port=8000)
