import streamlit as st
import pandas as pd
import joblib


#page layout
st.set_page_config( page_title= "Probability of Default System", page_icon="🏦", layout="centered")
st.title("🏦 CREDIT RISK ANALYSIS by Rinae Musindane")
st.subheader("Probability of Default Prediction")
st.write("Enter the applicant's information below.")

#model
import os
project_folder = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(project_folder, "credit_model.pkl"))
scaler = joblib.load(os.path.join(project_folder, "scaler.pkl"))
model_columns = joblib.load(os.path.join(project_folder, "model_columns.pkl"))

#inputs
person_age = st.number_input("Age[18,100]",min_value=18,max_value=100)

person_income = st.number_input("Annual Income", min_value=0.0,)

person_emp_length = st.number_input("Employment Length (Years)",min_value=0.0)

loan_amnt = st.number_input( "Loan Amount",min_value=0.0)

loan_int_rate = st.number_input( "Interest Rate (%)", min_value=0.0)

loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0,)

cb_person_cred_hist_length = st.number_input("Credit History Length",min_value=0)

person_home_ownership = st.selectbox("Home Ownership",["RENT", "OWN", "MORTGAGE", "OTHER"])

loan_intent = st.selectbox( "Loan Intent", ["EDUCATION","MEDICAL","PERSONAL","VENTURE","HOMEIMPROVEMENT", "DEBTCONSOLIDATION" ])

loan_grade = st.selectbox("Loan Grade",["A", "B", "C", "D", "E", "F", "G"])

cb_person_default_on_file = st.selectbox("Previous Default",["N", "Y"])

#prediction 

if st.button("Predict"):

    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length]
    })

    # Convert categorical variables
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Decision
    if probability < 0.50:
        decision = "✅ APPROVED"
    else:
        decision = "❌ REJECTED"

    #################################
    # RESULTS
    #################################

    st.divider()

    st.subheader("Prediction Result")

    st.metric("Probability of Default",f"{probability*100:.2f}%"  )

    st.metric("Decision",decision )

st.divider()
st.caption(   
    "🏦 Credit Risk Analysis Dashboard | "
    "Probability of Default (PD) Model | "
    "by RINAE MUSINDANE"
    )