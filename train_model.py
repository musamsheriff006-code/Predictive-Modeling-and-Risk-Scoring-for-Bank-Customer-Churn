import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# Load dataset
pd.read_csv("data/Churn_Modelling.csv")

# Drop unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Geography"] = le.fit_transform(data["Geography"])

# Split features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model (high accuracy)
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

print("Model Accuracy:", accuracy)
print("ROC AUC Score:", roc)

# Save model
model = joblib.load("models/churn_model.pkl")

print("Model saved successfully in models folder!")