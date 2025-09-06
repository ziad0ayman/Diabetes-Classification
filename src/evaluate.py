import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Load test data
DATA_PATH = os.path.join("data", "pima-indians-diabetes-database.csv")

def load_data(path=DATA_PATH):
    """
    Loads the Pima Indians Diabetes dataset from CSV.
    Assumes the last column is the target label.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)  # skip header
    X, y = data[:, :-1], data[:, -1]
    return X, y

def load_model():
    MODEL_PATH = os.path.join("model", "model.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate():
    # Load data and model
    X, y = load_data()
    model = load_model()

    # Predict
    y_pred = model.predict(X)

    # Compute metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print("📊 Model Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
