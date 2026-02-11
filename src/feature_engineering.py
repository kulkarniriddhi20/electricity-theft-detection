import pandas as pd
import numpy as np

# Load labeled data
data = pd.read_csv("data/processed/final_data.csv")

# Convert datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Create date and hour columns
data['date'] = data['datetime'].dt.date
data['hour'] = data['datetime'].dt.hour
data['weekday'] = data['datetime'].dt.weekday

# -------------------------------
# 1️⃣ Daily Mean Consumption
# -------------------------------
data['daily_mean'] = data.groupby(['customer_id', 'date'])['consumption'].transform('mean')

# -------------------------------
# 2️⃣ Daily Variance
# -------------------------------
data['daily_variance'] = data.groupby(['customer_id', 'date'])['consumption'].transform('var')

# -------------------------------
# 3️⃣ Night Consumption Ratio
# Night = 10 PM to 6 AM
# -------------------------------
night_data = data[(data['hour'] >= 22) | (data['hour'] <= 6)]
night_sum = night_data.groupby(['customer_id', 'date'])['consumption'].sum()

total_sum = data.groupby(['customer_id', 'date'])['consumption'].sum()

data['night_ratio'] = data.set_index(['customer_id', 'date']).index.map(
    (night_sum / total_sum).fillna(0)
)

# -------------------------------
# 4️⃣ Weekend Consumption Ratio
# -------------------------------
weekend_data = data[data['weekday'] >= 5]
weekend_sum = weekend_data.groupby(['customer_id', 'date'])['consumption'].sum()

data['weekend_ratio'] = data.set_index(['customer_id', 'date']).index.map(
    (weekend_sum / total_sum).fillna(0)
)

# -------------------------------
# 5️⃣ Sudden Drop Feature
# -------------------------------
data['prev_consumption'] = data.groupby('customer_id')['consumption'].shift(1)

data['sudden_drop'] = (data['prev_consumption'] - data['consumption']) / data['prev_consumption']
data['sudden_drop'] = data['sudden_drop'].fillna(0)

# Save features
features = data[['customer_id', 'date', 'daily_mean', 'daily_variance',
                 'night_ratio', 'weekend_ratio', 'sudden_drop', 'label']]

features.to_csv("data/processed/features.csv", index=False)

print("✅ Feature engineering completed")
print(features.head())
