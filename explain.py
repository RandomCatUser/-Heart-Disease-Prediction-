

import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils import load_model
from config import MODEL_PATH

# -----------------------------
# Load Model
# -----------------------------

model = load_model(MODEL_PATH)

# The pipeline contains preprocessing + classifier
preprocessor = model.named_steps["prep"]
classifier = model.named_steps["model"]

# -----------------------------
# Example Patient
# -----------------------------

patient = pd.DataFrame([{
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 220,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

# -----------------------------
# Transform Data
# -----------------------------

patient_processed = preprocessor.transform(patient)

# -----------------------------
# SHAP Explainer
# -----------------------------

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(patient_processed)

# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("\n==============================")
print("AI Explanation")
print("==============================")

print("Prediction:", "High Risk" if prediction else "Low Risk")
print(f"Probability: {probability:.2%}")

# -----------------------------
# Feature Names
# -----------------------------

feature_names = preprocessor.get_feature_names_out()

# -----------------------------
# SHAP Waterfall Plot
# -----------------------------

try:
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            data=patient_processed[0],
            feature_names=feature_names
        )
    )
except Exception:
    print("Waterfall plot not available for this model.")

# -----------------------------
# SHAP Bar Plot
# -----------------------------

try:
    shap.summary_plot(
        shap_values,
        patient_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()
    plt.show()

except Exception:
    print("Summary plot not available.")

# -----------------------------
# Feature Contributions
# -----------------------------

print("\nTop Feature Contributions:")

values = shap_values[1][0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": values
})

importance["Abs"] = importance["Contribution"].abs()

importance = importance.sort_values(
    "Abs",
    ascending=False
)

print(
    importance[
        ["Feature", "Contribution"]
    ].head(10)
)




