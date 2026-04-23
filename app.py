import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    clf_path = BASE_DIR / 'placement_prediction_pipeline.pkl'
    reg_path = BASE_DIR / 'salary_prediction_pipeline.pkl'
    
    classification_model = joblib.load(clf_path)
    regression_model = joblib.load(reg_path)
    return classification_model, regression_model

clf_model, reg_model = load_models()


def main():
    st.title("🎓 Student Placement & Salary Prediction App")

    st.write("Masukkan data mahasiswa untuk memprediksi placement dan estimasi salary.")

    gender = st.radio("Gender", ["Male", "Female"])
    ssc_percentage = st.number_input("SSC Percentage", 0, 100)
    hsc_percentage = st.number_input("HSC Percentage", 0, 100)
    degree_percentage = st.number_input("Degree Percentage", 0, 100)
    cgpa = st.number_input("CGPA", 0.0, 10.0)
    entrance_exam_score = st.number_input("Entrance Exam Score", 0, 100)
    technical_skill_score = st.number_input("Technical Skill Score", 0, 100)
    soft_skill_score = st.number_input("Soft Skill Score", 0, 100)
    internship_count = st.number_input("Internship Count", 0, 10)
    live_projects = st.number_input("Live Projects", 0, 10)
    work_experience_months = st.number_input("Work Experience (months)", 0, 60)
    certifications = st.number_input("Certifications", 0, 10)
    attendance_percentage = st.number_input("Attendance Percentage", 0, 100)
    backlogs = st.number_input("Backlogs", 0, 10)
    extracurricular_activities = st.radio("Extracurricular Activities", ["Yes", "No"])

    data = {
        'gender': str(gender), 
        'ssc_percentage': int(ssc_percentage),
        'hsc_percentage': int(hsc_percentage),  
        'degree_percentage': int(degree_percentage),
        'cgpa': float(cgpa),  
        'entrance_exam_score': int(entrance_exam_score),  
        'technical_skill_score': int(technical_skill_score),  
        'soft_skill_score': int(soft_skill_score),  
        'internship_count': int(internship_count),  
        'live_projects': int(live_projects),  
        'work_experience_months': int(work_experience_months),  
        'certifications': int(certifications),  
        'attendance_percentage': int(attendance_percentage),  
        'backlogs': int(backlogs),  
        'extracurricular_activities': str(extracurricular_activities)  
    }

    df = pd.DataFrame([data])

    if st.button("🚀 Make Prediction"):

        placement_pred = clf_model.predict(df)[0]

        salary_pred = reg_model.predict(df)[0]

        if placement_pred == 1:
            st.success("Student is likely to be placed")
            st.info(f"💰 Estimated Salary: {salary_pred:.2f} LPA")
        else:
            st.error("Student is not likely to be placed")


if __name__ == "__main__":
    main()
