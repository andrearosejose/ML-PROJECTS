from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("boston_model.pkl", "rb"))
scaler = pickle.load(open("boston_scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["CRIM"]),
            float(request.form["ZN"]),
            float(request.form["INDUS"]),
            float(request.form["CHAS"]),
            float(request.form["NOX"]),
            float(request.form["RM"]),
            float(request.form["AGE"]),
            float(request.form["DIS"]),
            float(request.form["RAD"]),
            float(request.form["TAX"]),
            float(request.form["PTRATIO"]),
            float(request.form["B"]),
            float(request.form["LSTAT"])
        ]

        final_features = scaler.transform([features])
        prediction = model.predict(final_features)[0]

        return render_template(
            "index.html",
            prediction_text=f"🏠 Estimated House Price: ${prediction:,.2f}"
        )

    except Exception:
        return render_template(
            "index.html",
            prediction_text="⚠️ Please enter valid numerical values"
        )

if __name__ == "__main__":
    app.run(debug=True)
