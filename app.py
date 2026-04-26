# STEP 1 — Import Libraries

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================
# STEP 2 — Load Datasets
# =========================================

fake_df = pd.read_csv("dataset/Fake.csv")
real_df = pd.read_csv("dataset/True.csv")


# =========================================
# STEP 3 — Add Labels
# =========================================

# Fake News = 0
fake_df["label"] = 0

# Real News = 1
real_df["label"] = 1


# =========================================
# STEP 4 — Combine Datasets
# =========================================

df = pd.concat([fake_df, real_df], ignore_index=True)


# =========================================
# STEP 5 — Remove Missing Values
# =========================================

df = df.dropna()


# =========================================
# STEP 6 — Clean Text Function
# =========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text


# Apply cleaning to text column

df["text"] = df["text"].apply(clean_text)


# =========================================
# STEP 7 — Convert Text to Numbers
# =========================================

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df["text"])


# =========================================
# STEP 8 — Labels
# =========================================

y = df["label"]


# =========================================
# STEP 9 — Split Dataset
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================
# STEP 10 — Train Machine Learning Model
# =========================================

model = LogisticRegression()

model.fit(X_train, y_train)


# =========================================
# STEP 11 — Test Accuracy
# =========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


# =========================================
# STEP 12 — Prediction Function
# =========================================

def predict_news(news):

    # Clean user input

    news = clean_text(news)

    # Convert text to vector

    news_vector = vectorizer.transform([news])

    # Predict

    prediction = model.predict(news_vector)

    return prediction[0]


# =========================================
# STEP 13 — User Input Loop
# =========================================

print("\n========== Fake News Detection System ==========\n")

while True:

    news = input("Enter news text (or type 'exit' to quit): ")

    if news.lower() == "exit":
        print("\nProgram Closed.")
        break

    result = predict_news(news)

    if result == 0:
        print("\nPrediction: Fake News\n")

    else:
        print("\nPrediction: Real News\n")

