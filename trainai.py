# train_ai.py
import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# =========================
# 1️⃣ Load and Clean Data
# =========================
data = pd.read_csv("data/aireviews.csv")

if "review_text" not in data.columns or "is_ai" not in data.columns:
    raise Exception("Dataset must have columns: review_text, is_ai")

# Basic cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["review_text"] = data["review_text"].apply(clean_text)

# =========================
# 2️⃣ Balance Dataset
# =========================
human = data[data["is_ai"] == 0]
ai = data[data["is_ai"] == 1]

if len(human) != len(ai):
    min_size = min(len(human), len(ai))
    human = resample(human, replace=False, n_samples=min_size, random_state=42)
    ai = resample(ai, replace=False, n_samples=min_size, random_state=42)

data_balanced = pd.concat([human, ai]).sample(frac=1, random_state=42)
print(f"✅ Dataset loaded successfully: {len(data_balanced)} samples")

X = data_balanced["review_text"]
y = data_balanced["is_ai"]

# =========================
# 3️⃣ Split + Vectorize
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 4️⃣ Train Both Models
# =========================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test_vec))

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_vec, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_vec))

# =========================
# 5️⃣ Compare + Save
# =========================
best_model = log_model if log_acc >= rf_acc else rf_model
best_name = "Logistic Regression" if log_acc >= rf_acc else "Random Forest"

os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/ai_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/ai_vectorizer.pkl", "wb"))

print("\n📊 Model Performance")
print(f"Logistic Regression: {log_acc*100:.2f}%")
print(f"Random Forest: {rf_acc*100:.2f}%")
print(f"🏆 Best Model: {best_name}")
print("✅ Model and vectorizer saved successfully.")
