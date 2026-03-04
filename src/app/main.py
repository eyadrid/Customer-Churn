import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.serving.inference import predict

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
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


# ─── Custom CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #F4F6FA;
    --surface:   #FFFFFF;
    --border:    #E2E8F0;
    --text:      #1A202C;
    --muted:     #64748B;
    --accent:    #0F62FE;
    --accent-lt: #EFF4FF;
    --danger:    #E53E3E;
    --danger-lt: #FFF5F5;
    --success:   #38A169;
    --success-lt:#F0FFF4;
    --warn:      #D97706;
    --warn-lt:   #FFFBEB;
    --radius:    12px;
    --shadow:    0 1px 3px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.06);
    --shadow-lg: 0 8px 30px rgba(0,0,0,.12);
}

/* ── Page base ── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── App header ── */
.app-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}
.app-header-icon {
    width: 44px; height: 44px;
    background: var(--accent);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 2px 0 0 !important;
}

/* ── Section cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px 18px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    transition: box-shadow .2s;
}
.section-card:hover { box-shadow: var(--shadow-lg); }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Gradio form controls ── */
.gradio-container label span {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: .5px;
}

select, input[type="number"], .gr-input, .gr-dropdown {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--text) !important;
    transition: border-color .15s, box-shadow .15s !important;
}
select:focus, input[type="number"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(15,98,254,.12) !important;
    outline: none !important;
}

/* ── Predict button ── */
button.gr-button-primary, .predict-btn button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: .3px;
    padding: 13px 32px !important;
    cursor: pointer !important;
    transition: background .18s, transform .12s, box-shadow .18s !important;
    box-shadow: 0 2px 8px rgba(15,98,254,.3) !important;
    width: 100% !important;
}
button.gr-button-primary:hover, .predict-btn button:hover {
    background: #0050D8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(15,98,254,.38) !important;
}
button.gr-button-primary:active, .predict-btn button:active {
    transform: translateY(0) !important;
}

/* ── Result panel ── */
.result-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 24px;
    box-shadow: var(--shadow);
    min-height: 200px;
}

/* ── Churn / Safe banners inside the output textbox ── */
.churn-result-churn textarea, .churn-result-churn input {
    background: var(--danger-lt) !important;
    border: 2px solid var(--danger) !important;
    color: var(--danger) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
}
.churn-result-safe textarea, .churn-result-safe input {
    background: var(--success-lt) !important;
    border: 2px solid var(--success) !important;
    color: var(--success) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
}

/* ── Risk indicator bar ── */
.risk-bar-wrap {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    box-shadow: var(--shadow);
}
.risk-bar-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 10px;
}
.risk-bar-track {
    height: 10px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 6px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .5s cubic-bezier(.4,0,.2,1), background .4s;
}

/* ── Examples panel ── */
.gr-samples-table {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    font-size: 0.8rem !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
    padding: 18px 0 8px;
    border-top: 1px solid var(--border);
    margin-top: 16px;
}

/* ── Numeric badge chips on stat cards ── */
.stat-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--accent-lt);
    color: var(--accent);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: .3px;
}

/* ── Smooth fade-in ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.gradio-container > * { animation: fadeUp .35s ease both; }
"""

# ─── Prediction logic ─────────────────────────────────────────────────────────

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract,
        "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "tenure": int(tenure), "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    result = predict(data)
    is_churn = "churn" in str(result).lower() and "not" not in str(result).lower()

    # Live feedback string with context
    if is_churn:
        risk_level = "HIGH"
        emoji = "🔴"
        verdict = "Likely to Churn"
        insight = _get_churn_insight(data)
        output = (
            f"{emoji}  RISK LEVEL: {risk_level}\n"
            f"Verdict: {verdict}\n\n"
            f"⚠  Key Risk Factors:\n{insight}"
        )
    else:
        risk_level = "LOW"
        emoji = "🟢"
        verdict = "Not Likely to Churn"
        insight = _get_safe_insight(data)
        output = (
            f"{emoji}  RISK LEVEL: {risk_level}\n"
            f"Verdict: {verdict}\n\n"
            f"✅  Retention Signals:\n{insight}"
        )

    return output


def _get_churn_insight(d):
    flags = []
    if d["Contract"] == "Month-to-month":
        flags.append("• Month-to-month contract (no long-term commitment)")
    if d["InternetService"] == "Fiber optic" and d["OnlineSecurity"] == "No":
        flags.append("• Fiber optic without security add-ons")
    if d["PaymentMethod"] == "Electronic check":
        flags.append("• Electronic check payment (higher churn correlation)")
    if d["tenure"] <= 6:
        flags.append(f"• Very short tenure ({d['tenure']} months)")
    if d["TechSupport"] == "No":
        flags.append("• No tech support subscription")
    if not flags:
        flags.append("• Profile matches historical churn patterns")
    return "\n".join(flags)


def _get_safe_insight(d):
    signals = []
    if d["Contract"] in ("One year", "Two year"):
        signals.append(f"• Long-term {d['Contract'].lower()} contract")
    if d["tenure"] >= 24:
        signals.append(f"• Established customer ({d['tenure']} months tenure)")
    if d["OnlineSecurity"] == "Yes" and d["TechSupport"] == "Yes":
        signals.append("• Subscribed to security & support services")
    if d["PaymentMethod"] in ("Bank transfer (automatic)", "Credit card (automatic)"):
        signals.append("• Automatic payment method (loyalty indicator)")
    if not signals:
        signals.append("• Profile aligns with retained customers")
    return "\n".join(signals)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
  <div class="app-header-icon">📡</div>
  <div>
    <h1>Churn Intelligence Dashboard</h1>
    <p>Telecom Customer Retention · Powered by XGBoost ML</p>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
    <span class="stat-chip">⚡ Real-time</span>
    <span class="stat-chip">18 Features</span>
  </div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="Churn Intelligence Dashboard") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=3):

            # Section 1: Customer Profile
            gr.HTML('<div class="section-card"><div class="section-label">👤 Customer Profile</div>')
            with gr.Row():
                gender   = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                partner  = gr.Dropdown(["Yes", "No"], label="Has Partner", value="No")
                deps     = gr.Dropdown(["Yes", "No"], label="Has Dependents", value="No")
            gr.HTML('</div>')

            # Section 2: Phone Services
            gr.HTML('<div class="section-card"><div class="section-label">📞 Phone Services</div>')
            with gr.Row():
                phone = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                multi = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            gr.HTML('</div>')

            # Section 3: Internet Services
            gr.HTML('<div class="section-card"><div class="section-label">🌐 Internet Services</div>')
            with gr.Row():
                internet  = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                sec       = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                backup    = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            with gr.Row():
                device    = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                tech      = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                tv        = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                movies    = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")
            gr.HTML('</div>')

            # Section 4: Billing & Contract
            gr.HTML('<div class="section-card"><div class="section-label">💳 Billing & Contract</div>')
            with gr.Row():
                contract  = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type", value="Month-to-month")
                paperless = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                payment   = gr.Dropdown([
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ], label="Payment Method", value="Electronic check")
            with gr.Row():
                tenure   = gr.Number(label="Tenure (months)", value=1,   minimum=0, maximum=100)
                monthly  = gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200)
                total    = gr.Number(label="Total Charges ($)",   value=85.0, minimum=0, maximum=10000)
            gr.HTML('</div>')

        # ── Right column: prediction panel ────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="result-panel">')
            gr.HTML('<div class="section-label" style="margin-bottom:18px;">📊 Prediction Result</div>')

            predict_btn = gr.Button("▶  Run Prediction", variant="primary", elem_classes=["predict-btn"])

            output_box = gr.Textbox(
                label="Analysis",
                lines=10,
                placeholder=(
                    "Click 'Run Prediction' to analyze this customer...\n\n"
                    "Results will include:\n"
                    "• Risk level (HIGH / LOW)\n"
                    "• Churn verdict\n"
                    "• Key contributing factors"
                ),
                show_copy_button=True,
            )
            gr.HTML('</div>')

            # Tips card
            gr.HTML("""
            <div style="background:var(--accent-lt);border:1px solid #C7D7FD;border-radius:var(--radius);
                        padding:16px 18px;margin-top:12px;">
              <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;
                          color:var(--accent);margin-bottom:8px;">💡 Analyst Tips</div>
              <ul style="font-size:.8rem;color:#1e40af;margin:0;padding-left:16px;line-height:1.8;">
                <li>Month-to-month + Fiber optic = highest risk combo</li>
                <li>Tenure &lt; 6 months signals early-life churn</li>
                <li>Auto-pay methods correlate with retention</li>
                <li>Security + TechSupport subscriptions reduce risk</li>
              </ul>
            </div>
            """)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.HTML('<div class="section-label" style="margin:24px 0 10px;">🧪 Quick Test Cases</div>')
    gr.Examples(
        examples=[
            ["Female", "No",  "No",  "Yes", "No",  "Fiber optic", "No",  "No",  "No",  "No",  "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",          1,  85.0,   85.0],
            ["Male",   "Yes", "Yes", "Yes", "Yes", "DSL",          "Yes", "Yes", "Yes", "Yes", "No",  "No",  "Two year",        "No",  "Credit card (automatic)",  60,  45.0, 2700.0],
            ["Female", "Yes", "No",  "Yes", "No",  "Fiber optic", "No",  "Yes", "No",  "No",  "Yes", "No",  "Month-to-month", "Yes", "Electronic check",          3,  79.5,  238.5],
        ],
        inputs=[
            gender, partner, deps, phone, multi, internet, sec, backup,
            device, tech, tv, movies, contract, paperless, payment,
            tenure, monthly, total
        ],
        label="Load Example Customer",
    )

    gr.HTML("""
    <div class="app-footer">
      Churn Intelligence Dashboard · XGBoost Model · Telecom Customer Analytics
    </div>
    """)

    all_inputs = [
        gender, partner, deps, phone, multi, internet, sec, backup,
        device, tech, tv, movies, contract, paperless, payment,
        tenure, monthly, total
    ]
    predict_btn.click(fn=gradio_interface, inputs=all_inputs, outputs=output_box)



app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    print("Gradio UI → http://127.0.0.1:8000/ui")
    uvicorn.run(app, host="127.0.0.1", port=8000)