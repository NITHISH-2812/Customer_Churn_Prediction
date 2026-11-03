
import os
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

# App title
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to leave the service "
    "using machine learning."
)

st.divider()

# MODEL PERFORMANCE DASHBOARD
st.header("📈 Model Performance")

st.caption(
    "Performance results from the latest training run "
    "using the test dataset."
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "80.38%")
col2.metric("Precision", "64.85%")
col3.metric("Recall", "57.22%")
col4.metric("F1-Score", "60.80%")
col5.metric("ROC-AUC", "83.59%")

st.subheader("Confusion Matrix")

confusion_data = pd.DataFrame(
    [[917, 116], [160, 214]],
    index=["Actual: Stay", "Actual: Churn"],
    columns=["Predicted: Stay", "Predicted: Churn"]
)

st.dataframe(confusion_data, width="stretch")
st.caption(
    "The model correctly identified 214 customers who churned. "
    "It missed 160 actual churners."
)

st.divider()

# CUSTOMER PREDICTION FORM
st.header("🔍 Predict Customer Churn")

st.write("Enter the customer's details below.")

with st.form("customer_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Has Partner?", ["Yes", "No"])

    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )

    phone = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

# PREDICTION RESULT
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(customer)[0][1]

        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )

        st.progress(float(probability))
