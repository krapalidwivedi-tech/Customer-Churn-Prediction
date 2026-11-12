import streamlit as st
import pickle
import pandas as pd

#LOAD FILES
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)


# STREAMLIT UI
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the company."
)

st.subheader("Enter Customer Details")


# CUSTOMER INPUTS
gender = st.selectbox(
    "Gender", ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen", ["Yes", "No"]
)

Partner = st.selectbox(
    "Partner", ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents", ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=1
)

PhoneService = st.selectbox(
    "Phone Service", ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=0.0
)


# 4. PREPROCESS INPUT DATA
def preprocess_input():

    data = {
        "gender": [gender],
        "SeniorCitizen": [
            1 if SeniorCitizen == "Yes" else 0
        ],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "Contract": [Contract],
        "TotalCharges": [TotalCharges]
    }

    df = pd.DataFrame(data)

    # Apply saved Label Encoders
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])

    # Maintain training feature order
    df = df[
        [
            'gender', 'SeniorCitizen', 'Partner',
            'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'Contract', 'TotalCharges'
        ]
    ]

    # Apply StandardScaler
    df_scaled = scaler.transform(df)

    return df_scaled


# 0PREDICTION
if st.button("Predict"):

    processed_data = preprocess_input()

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"🔴 High Risk of Churn "
            f"({probability * 100:.2f}%)"
        )

        st.write("### Tips to Prevent Churn")

        st.write("""
        - Improve customer communication
        - Offer personalized deals
        - Provide better service quality
        - Introduce loyalty programs
        - Monitor customer engagement
        - Resolve issues quickly
        - Offer incentives
        - Collect feedback regularly
        - Improve user experience
        - Stay competitive
        """)

    else:

        st.success(
            f"🟢 Low Risk of Churn "
            f"({(1 - probability) * 100:.2f}%)"
        )

        st.write("### Tips to Retain Customer")

        st.write("""
        - Maintain good service
        - Reward customer loyalty
        - Stay connected with customers
        - Provide consistent value
        - Keep improving services
        - Build strong relationships
        - Engage regularly
        - Simplify processes
        - Stay responsive
        - Appreciate customers
        """)