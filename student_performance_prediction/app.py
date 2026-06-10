import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Student Performance Predictor")

st.write("Enter student details below:")

# User Inputs
hours = st.slider("Study Hours", 0, 12, 5)

attendance = st.slider("Attendance Percentage", 0, 100, 75)

# Predict Button
if st.button("Predict Result"):

    input_data = np.array([[hours, attendance]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Student Will PASS")
    else:
        st.error("Student Will FAIL")