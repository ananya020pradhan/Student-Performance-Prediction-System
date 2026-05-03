import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_student

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.hero {
    padding: 40px;
    border-radius: 25px;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 900;
    color: white;
}

.hero p {
    font-size: 18px;
    color: #e0f2fe;
}

.kpi1 {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.kpi2 {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.kpi3 {
    background: linear-gradient(135deg, #f59e0b, #f97316);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.kpi4 {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.card1 {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.card2 {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.card3 {
    background: linear-gradient(135deg, #f97316, #ef4444);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.card4 {
    background: linear-gradient(135deg, #22c55e, #14b8a6);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.info {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    padding: 20px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 5px 18px rgba(0,0,0,0.25);
}

.success {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

.warning {
    background: linear-gradient(135deg, #ef4444, #f97316);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

.safe {
    background: linear-gradient(135deg, #22c55e, #4ade80);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
}

.risk {
    background: linear-gradient(135deg, #ef4444, #f97316);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #ec4899);
    color: white;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    border: none;
    padding: 12px 25px;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #4f46e5, #db2777);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b, #0f172a);
    color: white;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: white !important;
}

h1, h2, h3 {
    color: white;
}

label {
    color: white !important;
    font-weight: 600;
}

.stDataFrame {
    background-color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("🎓 Student Dashboard")

page = st.sidebar.radio(
    "📌 Navigate",
    [
        "🏠 Home",
        "🧑‍🎓 Single Prediction",
        "📂 Batch CSV Prediction",
        "📊 Data Visualization",
        "📘 Project Details"
    ]
)

# =========================
# Home Page
# =========================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Performance Prediction System</h1>
        <p>AI-powered academic risk detection system for early intervention and student success analytics</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown('<div class="kpi1"><h3>ML Task</h3><h2>Classification</h2></div>', unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="kpi2"><h3>Model</h3><h2>Random Forest</h2></div>', unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="kpi3"><h3>Output</h3><h2>Risk Score</h2></div>', unsafe_allow_html=True)

    with k4:
        st.markdown('<div class="kpi4"><h3>Use Case</h3><h2>EdTech</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="info">
    <h3>📌 Project Overview</h3>
    This system predicts whether a student is <b>On Track</b> or <b>At Risk</b>
    using academic and behavioral data such as attendance, quiz scores, assignments,
    midterm marks, study hours, LMS activity, previous GPA, and submission consistency.
    <br><br>
    It helps teachers, mentors, colleges, and EdTech platforms identify weak students early
    and provide personalized academic support.
    </div>
    """, unsafe_allow_html=True)

# =========================
# Single Prediction Page
# =========================
elif page == "🧑‍🎓 Single Prediction":
    st.title("🧑‍🎓 Single Student Prediction")

    st.markdown("""
    <div class="info">
    Enter one student's academic and behavioral details to predict whether the student is
    <b>On Track</b> or <b>At Risk</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        school_type = st.selectbox("School Type", ["Government", "Private"])
        parent_edu = st.selectbox("Parent Education", ["High School", "Graduate", "Post Graduate"])
        prior_gpa = st.slider("Previous GPA", 5.0, 10.0, 7.5)

    with col2:
        attendance_pct = st.slider("Attendance Percentage", 40.0, 100.0, 75.0)
        quiz_avg = st.slider("Quiz Average", 20.0, 100.0, 65.0)
        assign_avg = st.slider("Assignment Average", 30.0, 100.0, 70.0)
        midterm = st.slider("Midterm Score", 20.0, 100.0, 60.0)

    with col3:
        study_hours_wk = st.slider("Study Hours Per Week", 1.0, 20.0, 7.0)
        on_time_submit_pct = st.slider("On-time Submission Percentage", 30.0, 100.0, 75.0)
        lms_logins_wk = st.slider("LMS Logins Per Week", 0, 15, 5)
        forum_posts = st.slider("Forum Posts", 0, 10, 2)
        commute_min = st.slider("Commute Time in Minutes", 5, 120, 35)

    sample = {
        "gender": gender,
        "school_type": school_type,
        "prior_gpa": prior_gpa,
        "attendance_pct": attendance_pct,
        "quiz_avg": quiz_avg,
        "assign_avg": assign_avg,
        "midterm": midterm,
        "study_hours_wk": study_hours_wk,
        "on_time_submit_pct": on_time_submit_pct,
        "lms_logins_wk": lms_logins_wk,
        "forum_posts": forum_posts,
        "parent_edu": parent_edu,
        "commute_min": commute_min
    }

    if st.button("🚀 Predict Student Performance"):
        try:
            result = predict_student(sample)

            st.markdown("## 🎯 Prediction Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="kpi2">
                    <h3>Student Status</h3>
                    <h2>{result["status"]}</h2>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="kpi3">
                    <h3>Pass Probability</h3>
                    <h2>{result["pass_probability"]}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                if result["status"] == "At Risk":
                    st.markdown("""
                    <div class="risk">
                        <h3>⚠️ Risk Level</h3>
                        <h2>Needs Support</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe">
                        <h3>✅ Risk Level</h3>
                        <h2>Academically Stable</h2>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            if result["status"] == "At Risk":
                st.markdown("""
                <div class="warning">
                <h3>📢 Recommended Intervention Plan</h3>
                <ul>
                    <li>Schedule weekly mentor check-ins.</li>
                    <li>Improve attendance through monitoring.</li>
                    <li>Provide extra doubt-clearing sessions.</li>
                    <li>Increase quiz and assignment practice.</li>
                    <li>Create a personalized study routine.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success">
                <h3>🌟 Recommendation</h3>
                The student is performing well. Continue regular attendance, timely submissions,
                active learning, and consistent study habits.
                </div>
                """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("Model not found. Please run `python main.py` first.")

# =========================
# Batch CSV Prediction Page
# =========================
elif page == "📂 Batch CSV Prediction":
    st.title("📂 Batch Prediction Using CSV Upload")

    st.markdown("""
    <div class="info">
    Upload a CSV file containing multiple students' data.
    The dashboard will predict risk status for every student and generate a downloadable report.
    </div>
    """, unsafe_allow_html=True)

    required_columns = [
        "gender", "school_type", "prior_gpa", "attendance_pct", "quiz_avg",
        "assign_avg", "midterm", "study_hours_wk", "on_time_submit_pct",
        "lms_logins_wk", "forum_posts", "parent_edu", "commute_min"
    ]

    st.markdown("### Required CSV Columns")
    st.code(", ".join(required_columns))

    uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
            else:
                model = joblib.load("models/student_model.pkl")
                feature_columns = joblib.load("models/feature_columns.pkl")

                processed_df = pd.get_dummies(df)
                processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

                predictions = model.predict(processed_df)
                probabilities = model.predict_proba(processed_df)[:, 1]

                df["Prediction"] = ["On Track" if p == 1 else "At Risk" for p in predictions]
                df["Pass Probability (%)"] = (probabilities * 100).round(2)

                df["Suggested Intervention"] = df["Prediction"].apply(
                    lambda x: "Weekly mentoring, attendance improvement, extra practice"
                    if x == "At Risk"
                    else "Continue current academic routine"
                )

                total = len(df)
                risk_count = (df["Prediction"] == "At Risk").sum()
                safe_count = (df["Prediction"] == "On Track").sum()

                b1, b2, b3 = st.columns(3)

                with b1:
                    st.markdown(f'<div class="kpi1"><h3>Total Students</h3><h2>{total}</h2></div>', unsafe_allow_html=True)

                with b2:
                    st.markdown(f'<div class="kpi3"><h3>At Risk Students</h3><h2>{risk_count}</h2></div>', unsafe_allow_html=True)

                with b3:
                    st.markdown(f'<div class="kpi4"><h3>On Track Students</h3><h2>{safe_count}</h2></div>', unsafe_allow_html=True)

                st.markdown("### Prediction Results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Report",
                    data=csv,
                    file_name="student_prediction_results.csv",
                    mime="text/csv"
                )

        except FileNotFoundError:
            st.error("Model not found. Please run `python main.py` first.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# =========================
# Data Visualization Page
# =========================
elif page == "📊 Data Visualization":
    st.title("📊 Student Data Visualization Dashboard")

    try:
        df = pd.read_csv("data/raw/students.csv")

        st.markdown("""
        <div class="info">
        This section shows visual insights from the student dataset such as performance distribution,
        attendance impact, score patterns, and feature relationships.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f'<div class="kpi1"><h3>Total Students</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="kpi2"><h3>Avg Attendance</h3><h2>{df["attendance_pct"].mean():.2f}%</h2></div>', unsafe_allow_html=True)

        with c3:
            st.markdown(f'<div class="kpi3"><h3>Avg Final Score</h3><h2>{df["final_score"].mean():.2f}</h2></div>', unsafe_allow_html=True)

        with c4:
            st.markdown(f'<div class="kpi4"><h3>Pass Rate</h3><h2>{df["passed"].mean() * 100:.2f}%</h2></div>', unsafe_allow_html=True)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Pass vs At Risk Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="passed", palette=["#ef4444", "#22c55e"], ax=ax1)
            ax1.set_xticklabels(["At Risk", "On Track"])
            ax1.set_xlabel("Student Status")
            ax1.set_ylabel("Count")
            ax1.set_title("Student Performance Distribution")
            st.pyplot(fig1)

        with col_b:
            st.markdown("### School Type Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="school_type", palette=["#38bdf8", "#facc15"], ax=ax2)
            ax2.set_xlabel("School Type")
            ax2.set_ylabel("Count")
            ax2.set_title("Government vs Private Students")
            st.pyplot(fig2)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("### Attendance vs Final Score")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df,
                x="attendance_pct",
                y="final_score",
                hue="passed",
                palette=["#ef4444", "#22c55e"],
                ax=ax3
            )
            ax3.set_title("Attendance Impact on Final Score")
            ax3.set_xlabel("Attendance Percentage")
            ax3.set_ylabel("Final Score")
            st.pyplot(fig3)

        with col_d:
            st.markdown("### Study Hours vs Final Score")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df,
                x="study_hours_wk",
                y="final_score",
                hue="passed",
                palette=["#8b5cf6", "#f97316"],
                ax=ax4
            )
            ax4.set_title("Study Hours Impact on Final Score")
            ax4.set_xlabel("Study Hours Per Week")
            ax4.set_ylabel("Final Score")
            st.pyplot(fig4)

        st.markdown("### Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        fig5, ax5 = plt.subplots(figsize=(12, 7))
        sns.heatmap(numeric_df.corr(), cmap="Spectral", annot=False, ax=ax5)
        ax5.set_title("Correlation Between Numeric Features")
        st.pyplot(fig5)

        st.markdown("### Feature Distribution Explorer")

        selected_feature = st.selectbox(
            "Select Feature",
            [
                "attendance_pct",
                "quiz_avg",
                "assign_avg",
                "midterm",
                "study_hours_wk",
                "prior_gpa",
                "on_time_submit_pct",
                "lms_logins_wk",
                "commute_min"
            ]
        )

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_feature], kde=True, color="#ec4899", ax=ax6)
        ax6.set_title(f"Distribution of {selected_feature}")
        ax6.set_xlabel(selected_feature)
        ax6.set_ylabel("Frequency")
        st.pyplot(fig6)

    except FileNotFoundError:
        st.error("Dataset not found. Please run `python main.py` first.")

# =========================
# Project Details Page
# =========================
elif page == "📘 Project Details":
    st.title("📘 Project Details")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card1">
        <h3>🧠 Problem Statement</h3>
        <p>
        Educational institutions need early warning systems to identify students who may fail
        or underperform. This project predicts student academic risk using machine learning.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card2">
        <h3>🎯 Business Impact</h3>
        <p>
        The system helps teachers, mentors, and EdTech platforms provide early support,
        reduce failure rates, and improve student outcomes.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="card3">
        <h3>⚙️ ML Workflow</h3>
        <p>
        Student Data → Data Cleaning → Feature Engineering → Model Training →
        Evaluation → Prediction → Intervention Suggestion
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card4">
        <h3>🚀 Recruiter-Friendly Features</h3>
        <ul>
            <li>End-to-end machine learning pipeline</li>
            <li>Professional Streamlit dashboard</li>
            <li>Batch CSV prediction support</li>
            <li>FastAPI inference support</li>
            <li>Colorful data visualization dashboard</li>
            <li>Downloadable prediction reports</li>
            <li>Real-world education analytics use case</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:white;'>Built with Machine Learning, Python, Streamlit and FastAPI | Student Performance Prediction System</p>",
    unsafe_allow_html=True
)