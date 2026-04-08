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
    result.append({
        "customer_id": int(df.iloc[i]["customer_id"]) if "customer_id" in df.columns else i+1,
        "prediction": int(pred[i]),
        "probability": float(prob[i])
    })

# Print JSON for Node.js to read
print(json.dumps(result))