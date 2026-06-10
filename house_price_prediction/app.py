import streamlit as st
import pickle
import numpy as np

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------------
# Load Trained Model
# -----------------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #16a34a;
    color: white;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Title Section
# -----------------------------------

st.markdown(
    '<p class="title">🏠 House Price Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict House Prices Using Machine Learning</p>',
    unsafe_allow_html=True
)

# -----------------------------------
# Input Section
# -----------------------------------

st.write("## Enter House Details")

col1, col2 = st.columns(2)

with col1:
    size = st.number_input(
        "📏 House Size (sq ft)",
        min_value=500,
        max_value=5000,
        value=1200,
        step=100
    )

with col2:
    rooms = st.selectbox(
        "🛏 Number of Rooms",
        [1, 2, 3, 4, 5, 6]
    )

# -----------------------------------
# Location Selection
# -----------------------------------

location = st.radio(
    "📍 Select Location Quality",
    ["Poor", "Average", "Good", "Premium"],
    horizontal=True
)

# Convert text to numeric value

location_score = {
    "Poor": 3,
    "Average": 5,
    "Good": 7,
    "Premium": 10
}

location_value = location_score[location]

# -----------------------------------
# House Summary
# -----------------------------------

st.write("## House Summary")

summary1, summary2, summary3 = st.columns(3)

summary1.info(f"📏 {size} sq ft")
summary2.info(f"🛏 {rooms} Rooms")
summary3.info(f"📍 {location}")

# -----------------------------------
# Prediction Button
# -----------------------------------

if st.button("Predict House Price", key="predict_btn"):

    input_data = np.array([
        [size, rooms, location_value]
    ])

    prediction = model.predict(input_data)

    st.markdown(
        f"""
        <div class="result-box">
            Estimated House Price <br><br>
            ₹ {prediction[0]:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------
# Footer
# -----------------------------------

st.write("---")
st.caption("Built with Streamlit & Scikit-learn")