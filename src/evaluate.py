import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model():
    df = pd.read_csv("data/processed/cleaned_students.csv")

    X = df.drop(columns=["final_score", "passed"])
    y = df["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load("models/student_model.pkl")
    y_pred = model.predict(X_test)

    os.makedirs("outputs", exist_ok=True)

    report = classification_report(y_test, y_pred)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    importance = model.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature")
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    print("Evaluation completed. Outputs saved in outputs/ folder.")

if __name__ == "__main__":
    evaluate_model()