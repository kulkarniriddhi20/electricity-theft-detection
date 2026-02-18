import pandas as pd
import numpy as np

# --------------------------------------------------
# Load cleaned data
# --------------------------------------------------
data = pd.read_csv("data/processed/clean_data.csv")

# Convert datetime column
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort data
data = data.sort_values(['customer_id', 'datetime'])

# --------------------------------------------------
# Feature 1: Average Daily Consumption
# --------------------------------------------------
data['daily_mean'] = data.groupby('customer_id')['consumption'].transform('mean')

# --------------------------------------------------
# Feature 2: Night Usage Ratio (12 AM – 5 AM)
# --------------------------------------------------
data['hour'] = data['datetime'].dt.hour

night_usage = data[data['hour'].between(0, 5)].groupby('customer_id')['consumption'].sum()
total_usage = data.groupby('customer_id')['consumption'].sum()

data['night_ratio'] = data['customer_id'].map(night_usage / total_usage)
data['night_ratio'] = data['night_ratio'].fillna(0)

# --------------------------------------------------
# Feature 3: Weekend Usage Ratio
# --------------------------------------------------
data['day'] = data['datetime'].dt.dayofweek  # 5 & 6 = Weekend

weekend_usage = data[data['day'] >= 5].groupby('customer_id')['consumption'].sum()
data['weekend_ratio'] = data['customer_id'].map(weekend_usage / total_usage)
data['weekend_ratio'] = data['weekend_ratio'].fillna(0)

# --------------------------------------------------
# Feature 4: Consumption Variance
# --------------------------------------------------
data['variance'] = data.groupby('customer_id')['consumption'].transform('var')
data['variance'] = data['variance'].fillna(0)

# --------------------------------------------------
# Feature 5: Sudden Drop in Consumption
# --------------------------------------------------
data['prev_consumption'] = data.groupby('customer_id')['consumption'].shift(1)

data['sudden_drop'] = (
    (data['prev_consumption'] - data['consumption']) /
    data['prev_consumption']
)

data['sudden_drop'] = data['sudden_drop'].fillna(0)
data['sudden_drop'] = data['sudden_drop'].clip(lower=0)

# --------------------------------------------------
# Create Final Feature Dataset
# --------------------------------------------------
features = data[
    [
        'customer_id',
        'daily_mean',
        'night_ratio',
        'weekend_ratio',
        'variance',
        'sudden_drop'
    ]
]

# Remove duplicate rows per customer
features = features.drop_duplicates()

# Save features
features.to_csv("data/processed/features.csv", index=False)

print("✅ STEP 5 COMPLETED: Feature engineering successful")
print("📁 Saved file: data/processed/features.csv")
