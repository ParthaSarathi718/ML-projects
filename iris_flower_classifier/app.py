import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E3A2F;
    color: white;
    font-size: 24px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model
BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "iris_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Header
st.title("🌸 Iris Flower Classification App")

st.write(
    "Predict iris flower species using Machine Learning and KNN Classification."
)

# Sidebar
st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.5
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.0
)

petal_length = st.sidebar.slider(
    "Petal Length",
    1.0,
    7.0,
    4.0
)

petal_width = st.sidebar.slider(
    "Petal Width",
    0.1,
    3.0,
    1.2
)

# Metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Model Accuracy", "89%")

with col2:
    st.metric("Algorithm", "KNN")

# Prediction
if st.button("Predict Species"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    predicted_species = species[prediction[0]]

    # Styled result box
    st.markdown(
        f'''
        <div class="result-box">
        🌼 Predicted Species: {predicted_species}
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Show flower image
    
    image_path = BASE_DIR / "images"
    
    if predicted_species == "Setosa":
        st.image(str(image_path / "setosa.jpg"), width=300)
    
    elif predicted_species == "Versicolor":
        st.image(str(image_path / "versicolor.jpg"), width=300)
    
    else:
        st.image(str(image_path / "virginica.jpg"), width=300)

# Footer
st.markdown("---")

st.caption("Built with Python, Scikit-learn, and Streamlit")
