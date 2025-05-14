from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def clean_data(X, y):
    """ Drop rows with any missing values in either X or y """
    df = pd.concat([X, y], axis=1)
    df.dropna(inplace=True)
    y_clean = df[y.columns[0]]  
    X_clean = df.drop(columns=[y.columns[0]])
    return X_clean, y_clean


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"\n=== {model_name} ===")
    print(f"Training Accuracy:   {model.score(X_train, y_train):.3f}")
    print(f"Validation Accuracy: {acc:.3f}")
    print("Classification Report:\n", classification_report(y_val, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))
    return model


def main():
    print("Obtain data from UCI repo")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Grouping 1-4 in to single class, binary target classification
    y_binary = y.copy()
    y_binary[y_binary > 0] = 1
    
    y = y_binary
    print(y.value_counts())

    print("Cleaning data")
    X, y = clean_data(X, y)

    print("Splitting data")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # training SVM and RandomForestClassifier
    print("Training models")
    svm_model = train_and_evaluate(SVC(gamma='auto'), X_train, y_train, X_val, y_val, "SVM")
    rf_model = train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                                  X_train, y_train, X_val, y_val, "Random Forest")


if __name__ == "__main__":
    main()
