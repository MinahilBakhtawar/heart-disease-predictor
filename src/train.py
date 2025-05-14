from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_models(X_train, y_train):
    svm_model = SVC(gamma='auto')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return svm_model, rf_model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)