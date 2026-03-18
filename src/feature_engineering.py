# ---------------------------------------------
# electricity_theft_detection: feature_engineering.py
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("data/processed/clean_data.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.sort_values(['customer_id', 'datetime'])

# -----------------------------
# Feature Engineering
# -----------------------------

# 1. Basic Features
data['daily_mean'] = data.groupby('customer_id')['consumption'].transform('mean')
data['hour'] = data['datetime'].dt.hour
night_usage = data[data['hour'].between(0, 5)].groupby('customer_id')['consumption'].sum()
total_usage = data.groupby('customer_id')['consumption'].sum()
data['night_ratio'] = data['customer_id'].map(night_usage / total_usage).fillna(0)
data['day'] = data['datetime'].dt.dayofweek
weekend_usage = data[data['day'] >= 5].groupby('customer_id')['consumption'].sum()
data['weekend_ratio'] = data['customer_id'].map(weekend_usage / total_usage).fillna(0)
data['variance'] = data.groupby('customer_id')['consumption'].transform('var').fillna(0)
data['prev_consumption'] = data.groupby('customer_id')['consumption'].shift(1)
data['sudden_drop'] = ((data['prev_consumption'] - data['consumption']) / data['prev_consumption']).fillna(0).clip(lower=0)
data['load_factor'] = data['daily_mean'] / data.groupby('customer_id')['consumption'].transform('max')
data['load_factor'] = data['load_factor'].fillna(0)

# 2. Final Features
features = data[['customer_id', 'daily_mean', 'night_ratio', 'weekend_ratio', 'variance', 'sudden_drop', 'load_factor']].drop_duplicates()

# 3. Risk Score (optional)
features['risk_score'] = 0.3*features['night_ratio'] + 0.3*features['sudden_drop'] + 0.2*features['variance'] + 0.2*features['daily_mean']

# -----------------------------
# Create Target
# -----------------------------
# Detect high-risk using percentile of risk_score
percentile = 99  # top 1% as theft
threshold = np.percentile(features['risk_score'], percentile)
features['target'] = (features['risk_score'] >= threshold).astype(int)

print("\n📊 Target Distribution:")
print(features['target'].value_counts())

# -----------------------------
# Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split
X = features[['daily_mean','night_ratio','weekend_ratio','variance','sudden_drop','load_factor']]
y = features['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# Handle Imbalance with SMOTE
# -----------------------------
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("\n✅ After SMOTE, Target Distribution:")
print(pd.Series(y_train_res).value_counts())

# -----------------------------
# Train XGBoost Classifier
# -----------------------------
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# -----------------------------
# Predictions
# -----------------------------
y_prob = model.predict_proba(X_test)[:,1]

# Choose threshold for high recall
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, classification_report
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2*precision*recall/(precision+recall+1e-6)
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
# Feature Importance
# -----------------------------
import matplotlib.pyplot as plt
importance = model.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# Save Model
# -----------------------------
with open("models/theft_model_xgb.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved successfully as theft_model_xgb.pkl")