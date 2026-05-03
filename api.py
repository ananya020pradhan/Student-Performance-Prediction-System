from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_student

app = FastAPI(title="Student Performance Prediction API")

class StudentInput(BaseModel):
    gender: str
    school_type: str
    prior_gpa: float
    attendance_pct: float
    quiz_avg: float
    assign_avg: float
    midterm: float
    study_hours_wk: float
    on_time_submit_pct: float
    lms_logins_wk: int
    forum_posts: int
    parent_edu: str
    commute_min: int

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}

@app.post("/predict")
def predict(data: StudentInput):
    result = predict_student(data.dict())

    if result["status"] == "At Risk":
        intervention = "Improve attendance, attend doubt-clearing sessions, and follow a weekly study plan."
    else:
        intervention = "Student is performing well. Continue regular study and engagement."

    result["intervention"] = intervention
    return result