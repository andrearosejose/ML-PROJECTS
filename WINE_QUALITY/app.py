from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("wine_type_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulphates"]),
            float(request.form["alcohol"])
        ]

        final_features = scaler.transform([features])
        prediction = model.predict(final_features)

        result = "🍷 Red Wine" if prediction[0] == 1 else "🥂 White Wine"

        return render_template("index.html", prediction_text=f"Predicted Wine Type: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error in input values")

if __name__ == "__main__":
    app.run(debug=True)
