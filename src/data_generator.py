import pandas as pd
import numpy as np
import os

def generate_student_data(n=1000):
    np.random.seed(42)

    data = {
        "student_id": range(1, n + 1),
        "gender": np.random.choice(["Male", "Female"], n),
        "school_type": np.random.choice(["Government", "Private"], n),
        "prior_gpa": np.round(np.random.uniform(5.0, 10.0, n), 2),
        "attendance_pct": np.round(np.random.uniform(40, 100, n), 2),
        "quiz_avg": np.round(np.random.uniform(20, 100, n), 2),
        "assign_avg": np.round(np.random.uniform(30, 100, n), 2),
        "midterm": np.round(np.random.uniform(20, 100, n), 2),
        "study_hours_wk": np.round(np.random.uniform(1, 20, n), 2),
        "on_time_submit_pct": np.round(np.random.uniform(30, 100, n), 2),
        "lms_logins_wk": np.random.randint(0, 15, n),
        "forum_posts": np.random.randint(0, 10, n),
        "parent_edu": np.random.choice(["High School", "Graduate", "Post Graduate"], n),
        "commute_min": np.random.randint(5, 120, n)
    }

    df = pd.DataFrame(data)

    score = (
        df["prior_gpa"] * 8 +
        df["attendance_pct"] * 0.20 +
        df["quiz_avg"] * 0.20 +
        df["assign_avg"] * 0.15 +
        df["midterm"] * 0.25 +
        df["study_hours_wk"] * 1.5 +
        df["on_time_submit_pct"] * 0.10 +
        df["lms_logins_wk"] * 0.8 +
        df["forum_posts"] * 0.5 -
        df["commute_min"] * 0.05
    )

    df["final_score"] = np.round(score / 2.2, 2)
    df["passed"] = (df["final_score"] >= 50).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/students.csv", index=False)

    print("Dataset generated successfully: data/raw/students.csv")

if __name__ == "__main__":
    generate_student_data()