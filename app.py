import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# -------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------
st.set_page_config(page_title="AI Banking System", layout="wide")

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/churn_model.pkl")

# -------------------------
# LOAD DATA (DEFAULT + UPLOAD OPTION)
# -------------------------
data = None

try:
    data = pd.read_csv("Churn_Modelling.csv")
    st.success("✅ Default dataset loaded")
except:
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded dataset loaded")

# -------------------------
# DATA PREPROCESSING
# -------------------------
if data is not None:

    data.columns = data.columns.str.strip()

    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].astype(str).str.strip().str.capitalize()
        data["Gender"] = data["Gender"].map({"Male":1,"Female":0})

    if "Geography" in data.columns:
        data["Geography"] = data["Geography"].astype(str).str.strip()
        data["Geography"] = data["Geography"].map({
            "France":0,
            "Germany":1,
            "Spain":2,
            "India":3,
            "Australia":4
        })

# -------------------------
# 💱 CURRENCY SELECTOR
# -------------------------
st.sidebar.header("💱 Currency Settings")

currency = st.sidebar.selectbox(
    "Select Currency",
    ["₹ INR", "$ USD", "€ EUR", "£ GBP"]
)

symbols = {"₹ INR":"₹", "$ USD":"$", "€ EUR":"€", "£ GBP":"£"}
rates = {"₹ INR":1, "$ USD":0.012, "€ EUR":0.011, "£ GBP":0.0095}

symbol = symbols[currency]
rate = rates[currency]

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("🎛 Customer Inputs")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geo = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.slider("Num of Products", 1, 4, 2)
has_card = st.sidebar.selectbox("Has Credit Card", [0,1])
is_active = st.sidebar.selectbox("Is Active Member", [0,1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geo_map = {"France":0,"Germany":1,"Spain":2,"India":3,"Australia":4}
geo_val = geo_map[geo]
gender_val = 1 if gender=="Male" else 0

# -------------------------
# PREDICTION
# -------------------------
input_data = np.array([[

    credit_score,
    geo_val,
    gender_val,
    age,
    tenure,
    balance,
    products,
    has_card,
    is_active,
    salary

]])

prob = model.predict_proba(input_data)[0][1]
risk_score = int(prob * 100)

# -------------------------
# METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{prob:.2f}")
col2.metric("Risk Score", f"{risk_score}/100")

if prob > 0.6:
    col3.error("High Risk")
elif prob > 0.3:
    col3.warning("Medium Risk")
else:
    col3.success("Low Risk")

# -------------------------
# GAUGE
# -------------------------
st.subheader("🎯 Customer Risk Meter")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    gauge={
        'axis': {'range': [0,100]},
        'steps': [
            {'range': [0,30], 'color': "green"},
            {'range': [30,60], 'color': "yellow"},
            {'range': [60,100], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig_gauge, width="stretch")

# -------------------------
# RECOMMENDATIONS
# -------------------------
st.header("🤖 Retention Recommendations")

recs = []

if balance > 100000:
    recs.append("Offer premium investment plans")
if is_active == 0:
    recs.append("Engage customer with loyalty programs")
if products < 2:
    recs.append("Promote cross-selling")
if credit_score < 500:
    recs.append("Provide financial advisory")
if age > 60:
    recs.append("Offer retirement services")

if recs:
    for r in recs:
        st.write("✔️", r)
else:
    st.success("Customer is stable")

# -------------------------
# PIE CHART
# -------------------------
st.header("📈 Probability Distribution")

prob_df = pd.DataFrame({
    "Category":["Safe","Churn"],
    "Value":[1-prob, prob]
})

st.plotly_chart(px.pie(prob_df, values="Value", names="Category"), width="stretch")

# -------------------------
# DATA-DEPENDENT FEATURES
# -------------------------
if data is not None:

    # CLV
    st.header("💰 Customer Lifetime Value")
    data["CLV"] = data["Balance"] * data["Tenure"] * 0.1
    st.metric("Average CLV", f"{symbol}{data['CLV'].mean()*rate:,.0f}")

    st.plotly_chart(px.histogram(data, x="CLV"), width="stretch")

    # Segmentation
    st.header("👥 Customer Segmentation")

    features = data[["CreditScore","Balance","Age","EstimatedSalary"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Segment"] = kmeans.fit_predict(features)

    st.plotly_chart(px.scatter(data, x="Age", y="Balance", color="Segment"), width="stretch")

    # Insights
    st.header("📊 Advanced Insights")

    st.plotly_chart(px.histogram(data,x="Balance"), width="stretch")
    st.plotly_chart(px.histogram(data,x="CreditScore"), width="stretch")
    st.plotly_chart(px.scatter(data,x="Age",y="EstimatedSalary"), width="stretch")

else:
    st.warning("⚠️ Upload dataset to unlock analytics features")
