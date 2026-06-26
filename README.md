


# Heart Disease AI Prediction System

A machine learning system that predicts heart disease risk using clinical patient data.  
Built with Python, Scikit-learn, Streamlit, FastAPI, and SHAP.

> ⚠️ This project is for educational purposes only and is NOT a medical diagnostic tool.

---

## Features

- Multiple ML models (Random Forest, XGBoost, LightGBM, etc.)
- Automatic model selection
- Interactive Streamlit web app
- REST API using FastAPI
- Model explainability using SHAP
- Data visualization dashboard
- Risk probability scoring system

---

## Dataset

Uses the **UCI Heart Disease Dataset**:

Features include:
- Age
- Sex
- Chest pain type
- Blood pressure
- Cholesterol
- ECG results
- Max heart rate
- Exercise angina
- Oldpeak
- Thalassemia

Target:
- `0` = Low risk
- `1` = High risk

---

## Machine Learning Pipeline

1. Data preprocessing
2. Missing value handling
3. Encoding categorical variables
4. Feature scaling
5. Model training (10+ models)
6. Best model selection
7. Evaluation (Accuracy, ROC-AUC)
8. Model saving

---

## Project Structure

```bash
heart-disease-ai/
│
├── app.py              # Streamlit UI
├── api.py              # FastAPI backend
├── train.py            # Model training
├── predict.py          # CLI prediction
├── explain.py          # SHAP explainability
├── dashboard.py        # Analytics dashboard
├── utils.py
├── config.py
│
├── models/
│   └── model.pkl
│
├── data/
│   └── heart.csv
│
└── requirements.txt
````

---

##  Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train model

```bash
python train.py
```

---

### 3. Run Streamlit App

```bash
streamlit run app.py
```

---

### 4. Run Dashboard

```bash
streamlit run dashboard.py
```

---

### 5. Run API

```bash
uvicorn api:app --reload
```

---

### 6. CLI Prediction

```bash
python predict.py
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Example Request

```json
{
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
}
```

### Response

```json
{
  "prediction": "High Risk",
  "probability": 0.91,
  "confidence": 0.91,
  "recommendation": "Consult a healthcare professional promptly.",
  "disclaimer": "Not a medical diagnosis."
}
```

---

## Model Performance

* Accuracy: ~85%–95%
* ROC-AUC: ~0.88–0.96

(Depends on dataset split and model choice)

---

## Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost / LightGBM / CatBoost
* Streamlit
* FastAPI
* SHAP
* Matplotlib / Seaborn

---

## Disclaimer

This system is NOT a medical device.
It is intended only for educational and research purposes.

Always consult a certified medical professional.

---

## Author

Built as a machine learning project for educational purposes.

---

## Future Improvements

* Deep learning model (TensorFlow / PyTorch)
* Real hospital dataset integration
* Mobile app version
* Docker deployment
* Cloud hosting (AWS / GCP)

````

---

#  requirements.txt

```txt id="bq8zq1"
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
fastapi
uvicorn
joblib
plotly
shap
xgboost
lightgbm
catboost
````

---# -Heart-Disease-Prediction-
