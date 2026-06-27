import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import MODEL_PATH, TEST_SIZE, RANDOM_STATE, DATASET, UCI_DATA_URL, UCI_COLUMNS
from utils import save_model


def load_dataset(path=DATASET):
    if os.path.exists(path):
        print("Loaded local dataset:", path)
        return pd.read_csv(path)

    print("Downloading UCI Heart Disease dataset...")
    df = pd.read_csv(UCI_DATA_URL, names=UCI_COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset to {path} with {len(df)} rows.")
    return df


df = load_dataset()
print(df.head())

# -----------------------------
# Clean dataset (IMPORTANT)
# -----------------------------

df.replace("?", pd.NA, inplace=True)

df = df.apply(pd.to_numeric, errors="ignore")

df.dropna(inplace=True)

# -----------------------------
# Split features/target
# -----------------------------

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Feature types
# -----------------------------

categorical = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

categorical = [c for c in categorical if c in X.columns]

numeric = [c for c in X.columns if c not in categorical]

# -----------------------------
# Preprocessing pipelines
# -----------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical)
])

# -----------------------------
# Train/Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# Models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(n_estimators=200),

    "Decision Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(probability=True),

    "Neural Network": MLPClassifier(max_iter=1000),

    "XGBoost": XGBClassifier(eval_metric="logloss"),

    "LightGBM": LGBMClassifier(),

    "CatBoost": CatBoostClassifier(verbose=False)
}

# -----------------------------
# Train + Evaluate
# -----------------------------

best_model = None
best_score = 0

print("\n Model Performance:\n")

for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)

    try:
        probs = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0

    print(f"{name:20} Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")

    if acc > best_score:
        best_score = acc
        best_model = pipe

# -----------------------------
# Save best model
# -----------------------------

print("\n Best Accuracy:", best_score)

save_model(best_model, MODEL_PATH)

print(" Model saved to:", MODEL_PATH)