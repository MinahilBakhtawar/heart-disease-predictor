from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import save_report


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, report_path=None):
    print(f"\n=== {model_name} ===")
    train_acc = model.score(X_train, y_train)
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    report = classification_report(y_val, val_preds)
    cm = confusion_matrix(y_val, val_preds)

    print(f"Training Accuracy:   {train_acc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    if report_path:
        full_report = f"{model_name} Report\n\nAccuracy: {val_acc:.3f}\n\n{report}\n\nConfusion Matrix:\n{cm}"
        save_report(full_report, report_path)
