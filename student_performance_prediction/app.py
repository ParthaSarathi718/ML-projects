import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Get current folder path

BASE_DIR = Path(__file__).parent

# Load model

model_path = BASE_DIR / "model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Title

st.title("🎓 Student Performance Predictor v1")

st.write("Enter student details below:")

# Inputs

hours = st.slider("Study Hours", 0, 12, 5)

attendance = st.slider("Attendance Percentage", 0, 100, 75)

# Prediction

if st.button("Predict Result"):
    input_data = np.array([[hours, attendance]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Student Will PASS")
    else:
        st.error("Student Will FAIL")

