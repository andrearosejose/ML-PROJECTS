import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Predict the selling price of a car using ML")

@st.cache_resource
def load_artifacts():
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("fuel_encoder.pkl", "rb") as f:
        fuel_encoder = pickle.load(f)
    with open("transmission_encoder.pkl", "rb") as f:
        transmission_encoder = pickle.load(f)
    return model, scaler, fuel_encoder, transmission_encoder

model, scaler, fuel_encoder, transmission_encoder = load_artifacts()

year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0)

fuel_type = st.selectbox("Fuel Type", fuel_encoder.classes_)
transmission = st.selectbox("Transmission", transmission_encoder.classes_)
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

fuel_encoded = fuel_encoder.transform([fuel_type])[0]
transmission_encoded = transmission_encoder.transform([transmission])[0]
seller_individual = 1 if seller_type == "Individual" else 0

input_data = pd.DataFrame(
    np.zeros((1, scaler.n_features_in_)),
    columns=scaler.feature_names_in_
)

input_data.loc[0, "Year"] = year
input_data.loc[0, "Present_Price"] = present_price
input_data.loc[0, "Kms_Driven"] = kms_driven
input_data.loc[0, "Fuel_Type"] = fuel_encoded
input_data.loc[0, "Transmission"] = transmission_encoded

if "Seller_Type_Individual" in input_data.columns:
    input_data.loc[0, "Seller_Type_Individual"] = seller_individual

if st.button("Predict Selling Price"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
