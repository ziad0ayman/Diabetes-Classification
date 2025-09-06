import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from preprocess import preprocess

def train_model():

    X_train, X_test, y_train, y_test = preprocess()

    xgb_train = xgb.DMatrix(X_train, label = y_train)
    xgb_test = xgb.DMatrix(X_test, label = y_test)

    model_xgb = XGBClassifier(learning_rate=0.6, random_state=42)
    model_xgb.fit(X_train, y_train)



    # Save model
    with open("model/model.pkl", "wb") as file:
        pickle.dump(model_xgb, file)

if __name__ == "__main__":
    train_model()