import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# --- 1. Load the Model and Transformers ---
@st.cache_resource # Caches the model so it doesn't reload on every button click
def load_assets():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_assets()

# --- 2. Build the User Interface ---
st.title("📉 Customer Churn Prediction Engine")
st.markdown("Enter customer metrics below to evaluate retention probability.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Metrics")
    tenure = st.number_input("Tenure", min_value=0, max_value=120, value=12)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=1)
    payment_delay = st.number_input("Payment Delay", min_value=0, max_value=50, value=0)
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=15000.0, value=500.0)

with col2:
    st.subheader("Categorical Features")
    sub_type = st.selectbox("Subscription Type", encoders['Subscription Type'].classes_)
    contract_len = st.selectbox("Contract Length", encoders['Contract Length'].classes_)

# --- 3. Processing and Prediction Logic ---
if st.button("Analyze Risk", type="primary"):
    
    # Map the UI text back to numbers using the exact encoders from training
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'Support Calls': [support_calls],
        'Payment Delay': [payment_delay],
        'Total Spend': [total_spend],
        'Subscription Type': [encoders['Subscription Type'].transform([sub_type])[0]],
        'Contract Length': [encoders['Contract Length'].transform([contract_len])[0]]
    })
    
    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Execute inference
    prediction = model.predict(input_scaled)[0]
    probability = float(model.predict_proba(input_scaled)[0][1])
    
    # Display Results
    st.divider()
    if prediction == 1:
        st.error(f"🚨 **High Risk of Churn Detected**")
        st.write(f"**Probability:** {probability:.1%}")
        st.progress(probability)
        st.write("> *Action Required: Flag for immediate customer success intervention.*")
    else:
        st.success(f"✅ **Customer Stable**")
        st.write(f"**Churn Probability:** {probability:.1%}")
        st.progress(probability)