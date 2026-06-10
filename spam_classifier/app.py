import kagglehub
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download dataset
path = kagglehub.dataset_download(
    "uciml/sms-spam-collection-dataset"
)

# Load dataset
df = pd.read_csv(path + "/spam.csv", encoding='latin-1')

# Keep needed columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Convert text into numbers
cv = CountVectorizer()

X = cv.fit_transform(df['message'])

y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
msg = ["You won a free mobile phone"]

msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")