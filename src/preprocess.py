import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():

    data = pd.read_csv("../data/Churn_Modelling.csv")

    data = data.drop(["RowNumber","CustomerId","Surname"], axis=1)

    le = LabelEncoder()

    data["Gender"] = le.fit_transform(data["Gender"])
    data["Geography"] = le.fit_transform(data["Geography"])

    return data