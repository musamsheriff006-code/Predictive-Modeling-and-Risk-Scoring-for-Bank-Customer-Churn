import joblib
import numpy as np

model = joblib.load("../models/churn_model.pkl")

def predict_churn(data):

    data = np.array(data).reshape(1,-1)

    probability = model.predict_proba(data)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return probability,risk