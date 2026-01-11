from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Columns used during training
scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        Gender = 1 if request.form["Gender"] == "Male" else 0
        Married = 1 if request.form["Married"] == "Yes" else 0
        Education = 1 if request.form["Education"] == "Graduate" else 0
        Self_Employed = 1 if request.form["Self_Employed"] == "Yes" else 0

        Dependents = request.form["Dependents"]
        Dependents = 3 if Dependents == "3+" else int(Dependents)

        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = int(request.form["Credit_History"])

        Property_Area = request.form["Property_Area"]

        # Property Area One-Hot Encoding (drop_first=True)
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

        # Create input DataFrame
        input_data = pd.DataFrame([[
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History,
            Property_Area_Semiurban, Property_Area_Urban
        ]], columns=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History',
            'Property_Area_Semiurban', 'Property_Area_Urban'
        ])

        # Scale numeric columns
        input_data[scale_cols] = scaler.transform(input_data[scale_cols])

        # Prediction
        prediction = model.predict(input_data)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
