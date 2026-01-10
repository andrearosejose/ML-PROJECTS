import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


data = pd.read_csv("Car_price/car_data.csv")


def cap_outliers_iqr(df, col):
    q1, q3 = np.percentile(df[col], [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])


numeric_columns = ["Year", "Selling_Price", "Present_Price", "Kms_Driven"]

for col in numeric_columns:
    cap_outliers_iqr(data, col)


data.drop(columns=["Car_Name"], inplace=True)

fuel_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

data["Fuel_Type"] = fuel_encoder.fit_transform(data["Fuel_Type"])
data["Transmission"] = transmission_encoder.fit_transform(data["Transmission"])


data = pd.get_dummies(data, columns=["Seller_Type"], drop_first=True)


with open("Car_price/fuel_encoder.pkl", "wb") as f:
    pickle.dump(fuel_encoder, f)

with open("Car_price/transmission_encoder.pkl", "wb") as f:
    pickle.dump(transmission_encoder, f)


X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

with open("Car_price/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

with open("Car_price/car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)