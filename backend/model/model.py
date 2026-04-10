import sys
import os               # <-- you were missing this
import pandas as pd
import pickle
import json

# CSV file path passed from Node.js
file = sys.argv[1]

# Path relative to this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/theft_model_xgb.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load input CSV
df = pd.read_csv(file)

# Drop customer_id if exists
X = df.drop(columns=["customer_id"], errors="ignore")

# Make predictions
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

# Prepare JSON output

result = []
for i in range(len(df)):
    row = df.iloc[i]

    result.append({
        "customer_id": int(row["customer_id"]) if "customer_id" in df.columns else i+1,
        "prediction": int(pred[i]),
        "probability": float(prob[i]),

        # 👇 ADD THESE (for graph)
        "daily_mean": float(row["daily_mean"]) if "daily_mean" in df.columns else 0,
        "night_ratio": float(row["night_ratio"]) if "night_ratio" in df.columns else 0,
        "weekend_ratio": float(row["weekend_ratio"]) if "weekend_ratio" in df.columns else 0,
        "variance": float(row["variance"]) if "variance" in df.columns else 0
    })

# Print JSON for Node.js to read
print(json.dumps(result))