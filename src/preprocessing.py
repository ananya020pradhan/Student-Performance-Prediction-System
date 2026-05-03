import pandas as pd
import os

def preprocess_data():
    df = pd.read_csv("data/raw/students.csv")

    df.drop(columns=["student_id"], inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_students.csv", index=False)

    print("Data preprocessing completed: data/processed/cleaned_students.csv")

if __name__ == "__main__":
    preprocess_data()