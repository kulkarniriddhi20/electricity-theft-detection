# ---------------------------------------------
# electricity_theft_detection: step7_predict.py
# ---------------------------------------------

import pandas as pd
import pickle
import os

MODEL_PATH = "models/theft_model_xgb.pkl"
DATA_PATH = "data/processed/new_customers.csv"

# -----------------------------
# Load Model
# -----------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print("❌ Model file not found. Run train_xgboost.py first.")
    exit()

# -----------------------------
# Create sample file if missing
# -----------------------------
if not os.path.exists(DATA_PATH):
    print("⚠ new_customers.csv not found. Creating sample file...")

    sample_data = pd.DataFrame({
        "customer_id": [1001, 1002, 1003],
        "daily_mean": [15.2, 10.5, 20.1],
        "night_ratio": [0.05, 0.08, 0.03],
        "weekend_ratio": [0.12, 0.10, 0.15],
        "variance": [2.3, 1.5, 3.0],
        "sudden_drop": [0.01, 0.02, 0.00],
        "load_factor": [0.8, 0.6, 0.9]
    })

    os.makedirs("data/processed", exist_ok=True)
    sample_data.to_csv(DATA_PATH, index=False)

    print(f"✅ Sample file created at {DATA_PATH}")
    print("👉 Edit this file with real data and run again.")
    exit()

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv(DATA_PATH)

feature_cols = [
    'daily_mean',
    'night_ratio',
    'weekend_ratio',
    'variance',
    'sudden_drop',
    'load_factor'
]

# Check columns
missing_cols = [col for col in feature_cols if col not in data.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    exit()

X = data[feature_cols]

# -----------------------------
# Predict
# -----------------------------
data['theft_probability'] = model.predict_proba(X)[:, 1]
data['prediction'] = (data['theft_probability'] > 0.5).astype(int)

# -----------------------------
# Save Results
# -----------------------------
OUTPUT_PATH = "data/processed/predictions.csv"
data.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Prediction completed!")
print(data.head())
print(f"\n📁 Results saved at {OUTPUT_PATH}")