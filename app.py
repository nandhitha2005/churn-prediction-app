import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")
columns = joblib.load("model_columns.pkl")

def predict_fn(SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
               gender, Partner, Dependents, PhoneService,
               InternetService, Contract, PaymentMethod):

    # Validation
    if tenure < 0 or MonthlyCharges < 0 or TotalCharges < 0:
        return "❌ Invalid input values"

    input_dict = {col: 0 for col in columns}

    # Numeric
    input_dict["SeniorCitizen"] = SeniorCitizen
    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = MonthlyCharges
    input_dict["TotalCharges"] = TotalCharges

    # Categorical encoding
    if gender == "Male":
        input_dict["gender_Male"] = 1

    if Partner == "Yes":
        input_dict["Partner_Yes"] = 1

    if Dependents == "Yes":
        input_dict["Dependents_Yes"] = 1

    if PhoneService == "Yes":
        input_dict["PhoneService_Yes"] = 1

    if InternetService == "Fiber optic":
        input_dict["InternetService_Fiber optic"] = 1
    elif InternetService == "No":
        input_dict["InternetService_No"] = 1

    if Contract == "One year":
        input_dict["Contract_One year"] = 1
    elif Contract == "Two year":
        input_dict["Contract_Two year"] = 1

    if PaymentMethod == "Credit card":
        input_dict["PaymentMethod_Credit card (automatic)"] = 1
    elif PaymentMethod == "Electronic check":
        input_dict["PaymentMethod_Electronic check"] = 1
    elif PaymentMethod == "Mailed check":
        input_dict["PaymentMethod_Mailed check"] = 1

    df = pd.DataFrame([input_dict])
    df = df[columns]

    # Prediction
    proba = model.predict_proba(df)[0][1]

    # 🔥 Business-style output
    if proba > 0.7:
        return f"""
⚠️ High Risk of Churn ({proba*100:.2f}%)

Reason:
- Based on customer profile and usage patterns

Action:
- Offer discount or retention plan
"""
    elif proba > 0.5:
        return f"""
⚠️ Moderate Risk ({proba*100:.2f}%)

Action:
- Monitor customer behavior
"""
    else:
        return f"""
✅ Low Risk ({(1-proba)*100:.2f}%)

Action:
- Customer likely satisfied
"""


# Inputs
inputs = [
    gr.Dropdown([0,1], label="Senior Citizen", value=0),
    gr.Number(label="Tenure", value=12),
    gr.Number(label="Monthly Charges", value=50),
    gr.Number(label="Total Charges", value=600),
    gr.Dropdown(["Male", "Female"], value="Female", label="Gender"),
    gr.Dropdown(["Yes", "No"], value="Yes", label="Partner"),
    gr.Dropdown(["Yes", "No"], value="No", label="Dependents"),
    gr.Dropdown(["Yes", "No"], value="Yes", label="Phone Service"),
    gr.Dropdown(["DSL", "Fiber optic", "No"], value="DSL", label="Internet Service"),
    gr.Dropdown(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract"),
    gr.Dropdown(["Credit card", "Electronic check", "Mailed check"], value="Electronic check", label="Payment Method"),
]

# App UI
app = gr.Interface(
    fn=predict_fn,
    inputs=inputs,
    examples=[
        [0, 1, 80, 80, "Male", "No", "No", "Yes", "Fiber optic", "Month-to-month", "Electronic check"]
    ],
    outputs="text",
    title="Customer Churn Prediction System | ML Powered Retention Tool",
    description="This machine learning application predicts whether a telecom customer is likely to churn. It helps businesses identify at-risk customers and take proactive retention actions."
)

app.launch()