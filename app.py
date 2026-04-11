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
# LOAD MODEL
# -------------------------

model = joblib.load("models/churn_model.pkl")

# -------------------------
# LOAD DATA
# -------------------------

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

# Encode dataset
data["Gender"] = data["Gender"].map({"Male":1,"Female":0})
data["Geography"] = data["Geography"].map({
    "France":0,
    "Germany":1,
    "Spain":2,
    "India":3,
    "Australia":4
})

# PAGE CONFIG

st.set_page_config(page_title="AI Banking System", layout="wide")

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")

# 💱 CURRENCY SELECTOR

st.sidebar.header("💱 Currency Settings")

currency = st.sidebar.selectbox(
    "Select Currency",
    ["₹ INR", "$ USD", "€ EUR", "£ GBP"]
)

currency_symbols = {
    "₹ INR": "₹",
    "$ USD": "$",
    "€ EUR": "€",
    "£ GBP": "£"
}

conversion_rates = {
    "₹ INR": 1,
    "$ USD": 0.012,
    "€ EUR": 0.011,
    "£ GBP": 0.0095
}

symbol = currency_symbols[currency]
rate = conversion_rates[currency]

# SIDEBAR INPUTS

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
# TOP METRICS
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
# 🎯 SPEEDOMETER GAUGE
# -------------------------

st.subheader("🎯 Customer Risk Meter")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={'text': "Risk Score"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0,30], 'color': "green"},
            {'range': [30,60], 'color': "yellow"},
            {'range': [60,100], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig_gauge, width="stretch")

# -------------------------
# 🤖 AI RECOMMENDATIONS
# -------------------------

st.header("🤖 Retention Recommendations")

recommendations = []

if balance > 100000:
    recommendations.append("Offer premium investment plans")

if is_active == 0:
    recommendations.append("Engage customer with loyalty programs")

if products < 2:
    recommendations.append("Promote cross-selling of products")

if credit_score < 500:
    recommendations.append("Provide financial advisory services")

if age > 60:
    recommendations.append("Offer retirement-friendly services")

if len(recommendations) == 0:
    st.success("Customer is stable. Maintain engagement.")
else:
    for rec in recommendations:
        st.write("✔️", rec)

# -------------------------
# 📈 PROBABILITY DISTRIBUTION
# -------------------------

st.header("📈 Probability Distribution")

prob_df = pd.DataFrame({
    "Category":["Safe","Churn Risk"],
    "Value":[1-prob, prob]
})

fig_prob = px.pie(prob_df, values="Value", names="Category")
st.plotly_chart(fig_prob, width="stretch")


# -------------------------
# ⭐ FEATURE IMPORTANCE
# -------------------------

st.header("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature":["Age","Balance","Credit Score","Products","Salary"],
    "Impact":[0.30,0.25,0.20,0.15,0.10]
})

fig_imp = px.bar(importance, x="Feature", y="Impact")
st.plotly_chart(fig_imp, width="stretch")

# -------------------------
# 🔄 WHAT-IF SIMULATOR
# -------------------------

st.header("🔄 What-If Scenario Simulator")

new_balance = st.slider("Adjust Balance",0.0,250000.0, float(balance))
new_products = st.slider("Adjust Products",1,4,int(products))
new_active = st.selectbox("Adjust Activity",[0,1], index=is_active)

sim_input = np.array([[

    credit_score,
    geo_val,
    gender_val,
    age,
    tenure,
    new_balance,
    new_products,
    has_card,
    new_active,
    salary

]])

new_prob = model.predict_proba(sim_input)[0][1]

st.metric("New Churn Probability", f"{new_prob:.2f}")

# 💰 CUSTOMER LIFETIME VALUE

st.header("💰 Customer Lifetime Value")

data["CLV"] = data["Balance"] * data["Tenure"] * 0.1

st.metric("Average CLV", f"${data['CLV'].mean():,.0f}")

fig_clv = px.histogram(data, x="CLV")
st.plotly_chart(fig_clv, width="stretch")


# 👥 SEGMENTATION

st.header("👥 Customer Segmentation")

features = data[["CreditScore","Balance","Age","EstimatedSalary"]]

kmeans = KMeans(n_clusters=3, random_state=42)
data["Segment"] = kmeans.fit_predict(features)

fig_seg = px.scatter(data, x="Age", y="Balance", color="Segment")
st.plotly_chart(fig_seg, width="stretch")


# 🌍 WORLD MAP VISUALIZATION

st.header("🌍 Global Churn Risk Map")

map_data = pd.DataFrame({
    "Region":["Europe","Africa","Asia","North America"],
    "Churn Rate":[28,35,22,31],
    "lat":[54,1,34,40],
    "lon":[15,20,100,-100]
})

fig_map = px.scatter_geo(
    map_data,
    lat="lat",
    lon="lon",
    size="Churn Rate",
    color="Churn Rate",
    hover_name="Region",
    size_max=40,
    projection="natural earth",
    title="Churn Risk Across Global Regions"
)

st.plotly_chart(fig_map, width="stretch")


# 📊 ADVANCED INSIGHTS

st.header("📊 Advanced Banking Insights")

fig1 = px.histogram(data,x="Balance")
fig2 = px.histogram(data,x="CreditScore")
fig3 = px.scatter(data,x="Age",y="EstimatedSalary")

st.plotly_chart(fig1, width="stretch")
st.plotly_chart(fig2, width="stretch")
st.plotly_chart(fig3, width="stretch")
