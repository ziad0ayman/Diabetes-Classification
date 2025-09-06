import gradio as gr
import pickle
import numpy as np

MODEL_PATH = ("model\model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

def predict(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    """
    Takes a list of input features and returns whether the person is
    Diabetic or Not Diabetic.
    """
    X = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1, -1)
    prediction = model.predict(X)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Detector",
    description="Predicts whether a patient has diabetes based on input data."
)

if __name__ == "__main__":
    demo.launch()
