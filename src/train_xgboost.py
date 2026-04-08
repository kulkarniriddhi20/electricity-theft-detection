# ---------------------------------------------
# electricity_theft_detection: train_xgboost.py
# ---------------------------------------------

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Features
# -----------------------------
FEATURES_PATH = "data/processed/features.csv"

data = pd.read_csv(FEATURES_PATH)
print(f"✅ Loaded data from {FEATURES_PATH}")
print("Columns in dataset:", list(data.columns))

# -----------------------------
# Feature Columns
# -----------------------------
feature_cols = [
    'daily_mean',
    'night_ratio',
    'weekend_ratio',
    'variance',
    'sudden_drop',
    'load_factor'
]

# -----------------------------
# Create target if missing or invalid
# -----------------------------
if 'target' not in data.columns or data['target'].nunique() < 2:
    print("⚠ Creating target column...")

    for p in [95, 90, 85, 80]:
        threshold = np.percentile(data['risk_score'], p)
        data['target'] = (data['risk_score'] >= threshold).astype(int)

        if data['target'].nunique() > 1:
            print(f"✅ Target created using {p} percentile")
            break

    # Final fallback
    if data['target'].nunique() < 2:
        print("⚠ Warning: Still one class, applying random target")
        data['target'] = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])

# -----------------------------
# Clean Data
# -----------------------------
data = data.dropna(subset=feature_cols + ['target'])

print("\n📊 Target distribution:")
print(data['target'].value_counts())

# -----------------------------
# Train-Test Split
# -----------------------------
X = data[feature_cols]
y = data['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Safety check before SMOTE
if y_train.nunique() < 2:
    raise ValueError("❌ Training data has only one class. Check target creation.")

# -----------------------------
# SMOTE (Balancing)
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n✅ After SMOTE, Target Distribution:")
print(pd.Series(y_train_res).value_counts())

# -----------------------------
# Train Model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -----------------------------
# Predictions
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_pred = (y_prob >= best_threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
print("\n📊 MODEL RESULTS")
print(f"Threshold used: {best_threshold}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Feature Importance Graph
# -----------------------------
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(feature_cols, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# -----------------------------
# Save Model
# -----------------------------
MODEL_PATH = "models/theft_model_xgb.pkl"

os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved successfully at {MODEL_PATH}")

# -----------------------------
# Save Updated Dataset
# -----------------------------
data.to_csv(FEATURES_PATH, index=False)
print(f"✅ Features CSV updated at {FEATURES_PATH}")