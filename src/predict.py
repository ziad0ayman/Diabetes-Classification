import pickle
import numpy as np
import os

MODEL_PATH = os.path.join("model", "model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

def predict(input_data):
    """
    Takes a list of input features and returns whether the person is
    Diabetic or Not Diabetic.
    """
    X = np.array(input_data).reshape(1, -1)
    prediction = model.predict(X)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"
