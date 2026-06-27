import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config import DATASET

st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATASET)

df = load_data()

st.title("📊 Heart Disease AI Dashboard")

st.write("Dataset Overview")

st.dataframe(df.head())

# -------------------------
# Dataset Statistics
# -------------------------

st.header("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Patients", len(df))
col2.metric("Features", len(df.columns)-1)
col3.metric("Disease Rate",
            f"{df['target'].mean()*100:.1f}%")

# -------------------------
# Distribution
# -------------------------

st.header("Heart Disease Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    data=df,
    x="target",
    ax=ax
)

ax.set_xticklabels(["Low Risk","High Risk"])

st.pyplot(fig)

# -------------------------
# Correlation
# -------------------------

st.header("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# -------------------------
# Histograms
# -------------------------

st.header("Feature Distributions")

numeric_cols = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

feature = st.selectbox(
    "Select Feature",
    numeric_cols
)

fig, ax = plt.subplots(figsize=(8,4))

sns.histplot(
    df[feature],
    kde=True,
    ax=ax
)

st.pyplot(fig)

# -------------------------
# Boxplot
# -------------------------

st.header("Boxplot")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=df,
    x="target",
    y=feature,
    ax=ax
)

ax.set_xticklabels(["Low","High"])

st.pyplot(fig)

# -------------------------
# Pairplot
# -------------------------

st.header("Pairplot")

pair_features = [
    "age",
    "chol",
    "trestbps",
    "thalach",
    "target"
]

pair = sns.pairplot(
    df[pair_features],
    hue="target"
)

st.pyplot(pair.figure)

# -------------------------
# Feature Importance
# -------------------------

st.header("Feature Importance")

try:

    model = joblib.load("models/model.pkl")

    classifier = model.named_steps["model"]

    prep = model.named_steps["prep"]

    names = prep.get_feature_names_out()

    importance = classifier.feature_importances_

    imp = pd.DataFrame({
        "Feature": names,
        "Importance": importance
    })

    imp = imp.sort_values(
        "Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=imp.head(15),
        y="Feature",
        x="Importance",
        ax=ax
    )

    st.pyplot(fig)

except:

    st.warning(
        "Feature importance unavailable for current model."
    )

# -------------------------
# Dataset Table
# -------------------------

st.header("Complete Dataset")

st.dataframe(df)

# -------------------------
# Download
# -------------------------

csv = df.to_csv(index=False)

st.download_button(
    "Download Dataset",
    csv,
    file_name="heart.csv",
    mime="text/csv"
)