import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[["Hours", "Attendance"]]

# Target
y = data["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Trained Successfully")