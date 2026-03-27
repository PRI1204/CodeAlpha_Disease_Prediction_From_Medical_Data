"""
train_model.py  —  Breast Cancer Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains a Decision Tree Classifier on breast_cancer.csv and saves:
  ✅  model.pkl          — trained DecisionTreeClassifier (joblib)
  ✅  scaler.pkl         — fitted StandardScaler (joblib)
  ✅  model_columns.pkl  — ordered feature column list (joblib)

Usage:
    python train_model.py

Ensure breast_cancer.csv is in the SAME directory before running.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("📂  Loading breast_cancer.csv …")
df = pd.read_csv("breast_cancer.csv")
print(f"    Shape : {df.shape}")
print(f"    Benign (1) : {(df.target==1).sum()}   |   Malignant (0) : {(df.target==0).sum()}")

# ── 2. Features / Target ───────────────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

# ── 3. Train-Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊  Train : {len(X_train)} samples   |   Test : {len(X_test)} samples")

# ── 4. StandardScaler ─────────────────────────────────────────────────────────
print("⚖️   Fitting StandardScaler …")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. DecisionTreeClassifier ─────────────────────────────────────────────────
print("🌳  Training DecisionTreeClassifier  (entropy | max_depth=5) …")
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred       = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

test_acc  = accuracy_score(y_test,  y_pred)
train_acc = accuracy_score(y_train, y_pred_train)

print(f"\n📈  Train Accuracy : {train_acc * 100:.2f}%")
print(f"📈  Test  Accuracy : {test_acc  * 100:.2f}%")
print("\n📋  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
print("🧩  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 7. Save with joblib  (matches what app.py expects) ────────────────────────
print("\n💾  Saving artifacts with joblib …")
joblib.dump(model,     "model.pkl")
joblib.dump(scaler,    "scaler.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("\n🎉  All done!  Files created:")
print("     • model.pkl")
print("     • scaler.pkl")
print("     • model_columns.pkl")
print("\n▶️   Run the app:  streamlit run app.py")
