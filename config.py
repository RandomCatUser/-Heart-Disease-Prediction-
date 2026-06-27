MODEL_PATH = "models/model.pkl"

DATASET = "data/heart.csv"
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
UCI_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

RANDOM_STATE = 42

TEST_SIZE = 0.2