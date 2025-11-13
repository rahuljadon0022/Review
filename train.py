import pandas as pd
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🚀 Training Fake Review Detection Model...")

data = pd.read_csv("data/fake_reviews_dataset.csv")

# Standardize labels
data["label"] = data["label"].apply(
    lambda x: 1 if str(x).lower() in ["fake", "1", "f"] else 0
)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Logistic
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

best_model = log_model if log_acc >= rf_acc else rf_model
best_name = "Logistic Regression" if log_acc >= rf_acc else "Random Forest"

os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/fake_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/fake_vectorizer.pkl", "wb"))

print(f"✅ Best Fake Review Model: {best_name} ({max(log_acc, rf_acc)*100:.2f}%)")
