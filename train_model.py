
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("news_dataset.csv")

# Separate text and labels
X = df["text"]
y = df["label"]

# Create AI model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X, y)

# Save trained model
with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("AI model trained successfully!")
print("Model saved successfully!")
