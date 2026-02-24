import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("German Credit Risk Prediction")

st.write("Enter customer details below:")


duration = st.number_input("Duration (months)", min_value=1)
credit_amount = st.number_input("Credit Amount", min_value=0)
age = st.number_input("Age", min_value=18)

purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/tv", "education", "business"])
sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "rent", "free"])
job = st.selectbox("Job", ["unskilled", "skilled", "highly skilled"])
checking_status = st.selectbox("Checking Status", ["<0", "0<=X<200", ">=200", "no checking"])


if st.button("Predict Credit Risk"):

    user_data = {
        "duration": duration,
        "credit_amount": credit_amount,
        "age": age,
        "purpose": purpose,
        "sex": sex,
        "housing": housing,
        "job": job,
        "checking_status": checking_status
    }

    user_df = pd.DataFrame([user_data])

    user_df_encoded = pd.get_dummies(user_df)

    user_df_encoded = user_df_encoded.reindex(columns=model_columns, fill_value=0)

    user_scaled = scaler.transform(user_df_encoded)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    if prediction == 1:
        st.success(f"Good Credit Risk ✅ (Probability: {round(probability,3)})")
    else:
        st.error(f"Bad Credit Risk ❌ (Probability: {round(probability,3)})")