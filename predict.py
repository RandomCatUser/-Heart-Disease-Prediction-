"""
predict.py
-----------
Load the trained heart disease prediction model and predict
the risk for a new patient.
"""

import argparse
import pandas as pd
from utils import load_model
from config import MODEL_PATH


def predict(patient_data: dict):
    """
    Predict heart disease risk for a single patient.

    Parameters
    ----------
    patient_data : dict
        Dictionary containing patient features.

    Returns
    -------
    tuple
        (prediction, probability)
    """

    model = load_model(MODEL_PATH)

    df = pd.DataFrame([patient_data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


def print_report(prediction, probability):
    """Display prediction results nicely."""

    print("\n==============================")
    print(" Heart Disease Risk Prediction")
    print("==============================")

    print(f"Risk Probability : {probability:.2%}")

    if prediction == 1:
        print("Prediction       : HIGH RISK")
    else:
        print("Prediction       : LOW RISK")

    print("\nAI Recommendation")

    if probability >= 0.80:
        print("- Immediate medical evaluation is recommended.")
        print("- Consult a cardiologist.")
        print("- Monitor blood pressure and cholesterol.")
    elif probability >= 0.50:
        print("- Consider scheduling a medical check-up.")
        print("- Maintain a healthy lifestyle.")
    else:
        print("- Continue regular health monitoring.")
        print("- Maintain exercise and healthy eating.")

    print(
        "\nDISCLAIMER: This tool estimates heart disease risk "
        "for educational purposes only. It is NOT a medical "
        "diagnosis and should not replace professional medical advice."
    )


def interactive_mode():
    """Collect patient information interactively."""

    print("\nEnter Patient Information\n")

    patient = {
        "age": int(input("Age: ")),
        "sex": int(input("Sex (1=Male, 0=Female): ")),
        "cp": int(input("Chest Pain Type (0-3): ")),
        "trestbps": float(input("Resting Blood Pressure: ")),
        "chol": float(input("Cholesterol: ")),
        "fbs": int(input("Fasting Blood Sugar (>120 mg/dl, 1=True, 0=False): ")),
        "restecg": int(input("Resting ECG (0-2): ")),
        "thalach": float(input("Maximum Heart Rate Achieved: ")),
        "exang": int(input("Exercise Induced Angina (1=Yes, 0=No): ")),
        "oldpeak": float(input("Oldpeak: ")),
        "slope": int(input("Slope (0-2): ")),
        "ca": int(input("Number of Major Vessels (0-4): ")),
        "thal": int(input("Thal (0-3): "))
    }

    prediction, probability = predict(patient)

    print_report(prediction, probability)


def cli_mode():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Heart Disease Prediction"
    )

    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--trestbps", type=float, required=True)
    parser.add_argument("--chol", type=float, required=True)
    parser.add_argument("--fbs", type=int, required=True)
    parser.add_argument("--restecg", type=int, required=True)
    parser.add_argument("--thalach", type=float, required=True)
    parser.add_argument("--exang", type=int, required=True)
    parser.add_argument("--oldpeak", type=float, required=True)
    parser.add_argument("--slope", type=int, required=True)
    parser.add_argument("--ca", type=int, required=True)
    parser.add_argument("--thal", type=int, required=True)

    args = parser.parse_args()

    patient = vars(args)

    prediction, probability = predict(patient)

    print_report(prediction, probability)


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 1:
        interactive_mode()
    else:
        cli_mode()