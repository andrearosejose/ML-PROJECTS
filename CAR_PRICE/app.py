from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("car_price_model.pkl", "rb"))
fuel_encoder = pickle.load(open("fuel_encoder.pkl", "rb"))
transmission_encoder = pickle.load(open("transmission_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        Year = int(request.form["Year"])
        Present_Price = float(request.form["Present_Price"])
        Kms_Driven = int(request.form["Kms_Driven"])
        Owner = int(request.form["Owner"])

        Fuel_Type = request.form["Fuel_Type"]
        Transmission = request.form["Transmission"]
        Seller_Type = request.form["Seller_Type"]

        fuel_val = fuel_encoder.transform([Fuel_Type])[0]
        transmission_val = transmission_encoder.transform([Transmission])[0]

        seller_individual = 1 if Seller_Type == "Individual" else 0

        features = np.array([[Year, Present_Price, Kms_Driven, fuel_val, transmission_val, Owner, seller_individual]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Car Price: ₹ {round(prediction, 2)} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "main":
    app.run(host="0.0.0.0", port=5000, debug=True)