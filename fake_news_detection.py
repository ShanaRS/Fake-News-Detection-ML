import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===========================
# Load Datasets
# ===========================

fake_data = pd.read_csv("dataset/Fake.csv")
true_data = pd.read_csv("dataset/True.csv")

print("Fake Dataset Shape:", fake_data.shape)
print("True Dataset Shape:", true_data.shape)

# ===========================
# Add Labels
# Fake = 0
# Real = 1
# ===========================

fake_data["class"] = 0
true_data["class"] = 1

# ===========================
# Merge Datasets
# ===========================

data = pd.concat([fake_data, true_data], axis=0)

# Shuffle the dataset

data = data.sample(frac=1, random_state=42)

# Reset Index

data.reset_index(drop=True, inplace=True)

# ===========================
# Remove Unnecessary Columns
# ===========================

data = data.drop(["title", "subject", "date"], axis=1)

# ===========================
# Check Missing Values
# ===========================

print("\nMissing Values:")
print(data.isnull().sum())

# ===========================
# Text Cleaning Function
# ===========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'\w*\d\w*', '', text)

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    return text

# Apply Cleaning

data["text"] = data["text"].apply(clean_text)

# ===========================
# Features and Labels
# ===========================

X = data["text"]
y = data["class"]

# ===========================
# Split Dataset
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ===========================
# TF-IDF Vectorization
# ===========================

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print("\nVectorization Completed!")

# ===========================
# Logistic Regression
# ===========================

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")

print("Accuracy:", accuracy_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

# ===========================
# Decision Tree
# ===========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n===== Decision Tree =====")

print("Accuracy:", accuracy_score(y_test, dt_pred))

print(classification_report(y_test, dt_pred))

# ===========================
# Random Forest
# ===========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")

print("Accuracy:", accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

# ===========================
# Gradient Boosting
# ===========================

gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

print("\n===== Gradient Boosting =====")

print("Accuracy:", accuracy_score(y_test, gb_pred))

print(classification_report(y_test, gb_pred))

# ===========================
# Manual Testing Function
# ===========================

def predict_news(news):

    news = clean_text(news)

    vector = vectorizer.transform([news])

    print("\nPrediction Results")

    print("------------------------")

    print("Logistic Regression :", "Real News" if lr.predict(vector)[0] == 1 else "Fake News")

    print("Decision Tree       :", "Real News" if dt.predict(vector)[0] == 1 else "Fake News")

    print("Random Forest       :", "Real News" if rf.predict(vector)[0] == 1 else "Fake News")

    print("Gradient Boosting   :", "Real News" if gb.predict(vector)[0] == 1 else "Fake News")


# ===========================
# Manual Input
# ===========================

news = input("\nEnter News Article:\n")

predict_news(news)