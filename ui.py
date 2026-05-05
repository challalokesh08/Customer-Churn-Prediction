import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('churn_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict the probability of them leaving.")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict Churn Risk"):
    input_dict = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'gender_Female': 1 if gender == "Female" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'MultipleLines_No': 1 if multiple_lines == "No" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_DSL': 1 if internet_service == "DSL" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No': 1 if online_security == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No': 1 if online_backup == "No" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No': 1 if device_protection == "No" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No': 1 if tech_support == "No" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No': 1 if streaming_tv == "No" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No': 1 if streaming_movies == "No" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    expected_cols = scaler.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    confidence = prediction[0][0]
    if confidence > 0.5:
        st.error(f"Churn Risk: {confidence*100:.2f}%")
    else:
        st.success(f"Retention Likelihood: {(1-confidence)*100:.2f}%")