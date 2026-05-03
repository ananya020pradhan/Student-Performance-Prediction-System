import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    df = pd.read_csv("data/processed/cleaned_students.csv")

    X = df.drop(columns=["final_score", "passed"])
    y = df["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/student_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    print(f"Model trained successfully.")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_model()