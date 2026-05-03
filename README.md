# 🎓 Student Performance Prediction System

🚀 **An AI-powered machine learning system that predicts student academic performance and identifies at-risk students for early intervention.**

---

## 📌 Project Overview

This project builds an **end-to-end machine learning pipeline** to analyze student data and predict whether a student is:

* ✅ **On Track**
* ⚠️ **At Risk**

The system uses academic and behavioral features such as:

* Attendance
* Quiz & assignment scores
* Study hours
* LMS activity
* Previous GPA
* Submission consistency

👉 This helps **teachers, mentors, and EdTech platforms** take **early action** to improve student outcomes.

---

## 🎯 Key Features

✔️ Single student prediction (interactive dashboard)
✔️ Batch prediction using CSV upload
✔️ Risk probability scoring
✔️ Automated intervention suggestions
✔️ Downloadable prediction reports
✔️ Data visualization dashboard
✔️ Clean and professional UI (Streamlit)
✔️ FastAPI backend for real-time prediction

---

## 🧠 Machine Learning Workflow

```
Student Data → Data Cleaning → Feature Engineering → Model Training
→ Evaluation → Prediction → Risk Analysis → Intervention Suggestion
```

---

## ⚙️ Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Model:** Random Forest Classifier
* **Deployment Ready:** Yes

---

## 📂 Project Structure

```
Student-Performance-Prediction-System/
│
├── data/
├── models/
├── outputs/
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── api.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/ananya020pradhan/student-performance-prediction.git
cd student-performance-prediction
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Full Pipeline

```
python main.py
```

👉 This will:

* Generate dataset
* Train model
* Save model
* Generate evaluation outputs

---

## 💻 Run Dashboard

```
streamlit run app.py
```

---

## 🌐 Run FastAPI

```
uvicorn api:app --reload
```

Open in browser:
👉 http://127.0.0.1:8000/docs

---

## 📊 Demo (CSV Upload)

Upload a CSV file with columns:

```
gender,school_type,prior_gpa,attendance_pct,quiz_avg,assign_avg,midterm,
study_hours_wk,on_time_submit_pct,lms_logins_wk,forum_posts,parent_edu,commute_min
```

---

## 📈 Real-World Applications

* 🎓 Universities – Student performance monitoring
* 📚 Coaching centers – Weak student detection
* 🌐 EdTech platforms – Personalized learning
* 🏢 Corporate training – Performance tracking

---

## 💡 Future Improvements

* Add Deep Learning models
* Real-time student tracking system
* SHAP explainability
* Database integration
* Deployment on cloud


## 👩‍💻 Author

**Ananya Pradhan**

📍 B.Tech IT Student

👉 Feel free to connect with me on LinkedIn
www.linkedin.com/in/ananya-pradhan-10bb462ba

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
