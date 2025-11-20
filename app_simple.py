import streamlit as st
import pandas as pd
import joblib
import numpy as np                                                              

model = joblib.load("student_model.pkl")
scaler = joblib.load("student_scaler.pkl")
feature_order = joblib.load("student_feature_order.pkl")
numeric_cols = joblib.load("student_numeric_columns.pkl")

st.title("Student Performance Predictor")
st.write("Enter student details to predict final score")
  
col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Study_Hours_per_Week = st.slider("Study Hours per Week", 0, 40, 10)
    Attendance_Percentage = st.slider("Attendance (%)", 0, 100, 75)
    Previous_Sem_Score = st.slider("Previous Semester Score", 0, 100, 65)
    Parental_Education = st.selectbox("Parental Education", ["Low", "Medium", "High"])
    Internet_Access = st.selectbox("Internet Access", ["No", "Yes"])
    Family_Income = st.number_input("Family Income (monthly)", 0, 100000, 30000)
    Tutoring_Classes = st.selectbox("Tutoring Classes", ["No", "Yes"])
    Sports_Activity = st.selectbox("Sports Activity", ["No", "Yes"])

with col2:
    Extra_Curricular = st.selectbox("Extra Curricular", ["No", "Yes"])
    School_Type = st.selectbox("School Type", ["Government", "Private"])
    Sleep_Hours = st.slider("Sleep Hours per Day", 3, 10, 7)
    Travel_Time = st.number_input("Travel Time (minutes)", 0, 180, 30)
    Test_Anxiety_Level = st.slider("Test Anxiety (1-10)", 1, 10, 5)
    Peer_Influence = st.slider("Peer Influence (1-10)", 1, 10, 5)
    Teacher_Feedback = st.slider("Teacher Feedback (1-10)", 1, 10, 6)
    Motivation_Level = st.slider("Motivation (1-10)", 1, 10, 6)
    Library_Usage_per_Week = st.slider("Library Usage per Week", 0, 10, 3)

data = pd.DataFrame([{
    'Gender': 1 if Gender == "Male" else 0,
    'Study_Hours_per_Week': Study_Hours_per_Week,
    'Attendance_Percentage': Attendance_Percentage,
    'Previous_Sem_Score': Previous_Sem_Score,
    'Parental_Education': ["Low", "Medium", "High"].index(Parental_Education),
    'Internet_Access': 1 if Internet_Access == "Yes" else 0,
    'Family_Income': Family_Income,
    'Tutoring_Classes': 1 if Tutoring_Classes == "Yes" else 0,
    'Sports_Activity': 1 if Sports_Activity == "Yes" else 0,
    'Extra_Curricular': 1 if Extra_Curricular == "Yes" else 0,
    'School_Type': 1 if School_Type == "Private" else 0,
    'Sleep_Hours': Sleep_Hours,
    'Travel_Time': Travel_Time,
    'Test_Anxiety_Level': Test_Anxiety_Level,
    'Peer_Influence': Peer_Influence,
    'Teacher_Feedback': Teacher_Feedback,
    'Motivation_Level': Motivation_Level,
    'Library_Usage_per_Week': Library_Usage_per_Week
}])

if st.button("Predict Final Score"):
    data_ordered = data[feature_order]
    data_scaled = data_ordered.copy()
    data_scaled[numeric_cols] = scaler.transform(data_ordered[numeric_cols].values)
    prediction = model.predict(data_scaled.values)[0]
    st.success(f"Predicted Final Score: {prediction:.2f} / 100")                           
