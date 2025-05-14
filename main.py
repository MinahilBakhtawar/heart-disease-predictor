from sklearn.model_selection import train_test_split
from src.data_loader import load_heart_disease_data
from src.train import train_models, save_model
from src.evaluate import evaluate_model



def main():
    print("Loading data")
    X, y = load_heart_disease_data(binary=True)

    print("Splitting")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training models")
    svm_model, rf_model = train_models(X_train, y_train)

    print("Evaluating models")
    evaluate_model(svm_model, X_train, y_train, X_val, y_val,
                   "SVM", "outputs/classification_reports/svm_report.txt")
    evaluate_model(rf_model, X_train, y_train, X_val, y_val,
                   "Random Forest", "outputs/classification_reports/rf_report.txt")
    save_model(svm_model, "models/svm_model.pkl")
    save_model(rf_model, "models/rf_model.pkl")


if __name__ == "__main__":
    main()
