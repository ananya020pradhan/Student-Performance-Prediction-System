import pandas as pd
import joblib

def predict_student(sample_data):
    model = joblib.load("models/student_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    df = pd.DataFrame([sample_data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    status = "On Track" if prediction == 1 else "At Risk"

    return {
        "prediction": int(prediction),
        "status": status,
        "pass_probability": round(probability * 100, 2)
    }