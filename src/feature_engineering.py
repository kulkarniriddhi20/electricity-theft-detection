import pandas as pd
import numpy as np

# --------------------------------------------------
# Load cleaned data
# --------------------------------------------------
data = pd.read_csv("data/processed/clean_data.csv")
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort data
data = data.sort_values(['customer_id', 'datetime'])

# --------------------------------------------------
# BASIC FEATURES
# --------------------------------------------------

# Average daily consumption
data['daily_mean'] = data.groupby('customer_id')['consumption'].transform('mean')

# Night usage ratio (12 AM – 5 AM)
data['hour'] = data['datetime'].dt.hour
night_usage = data[data['hour'].between(0, 5)].groupby('customer_id')['consumption'].sum()
total_usage = data.groupby('customer_id')['consumption'].sum()
data['night_ratio'] = data['customer_id'].map(night_usage / total_usage).fillna(0)

# Weekend usage ratio
data['day'] = data['datetime'].dt.dayofweek
weekend_usage = data[data['day'] >= 5].groupby('customer_id')['consumption'].sum()
data['weekend_ratio'] = data['customer_id'].map(weekend_usage / total_usage).fillna(0)

# Consumption variance
data['variance'] = data.groupby('customer_id')['consumption'].transform('var').fillna(0)

# Sudden drop detection
data['prev_consumption'] = data.groupby('customer_id')['consumption'].shift(1)
data['sudden_drop'] = (
    (data['prev_consumption'] - data['consumption']) / data['prev_consumption']
).fillna(0).clip(lower=0)

# --------------------------------------------------
# 🔥 ADVANCED FEATURES
# --------------------------------------------------

# 1. Seasonal Consumption Pattern
data['month'] = data['datetime'].dt.month

season_map = {
    12: 'winter', 1: 'winter', 2: 'winter',
    3: 'summer', 4: 'summer', 5: 'summer',
    6: 'monsoon', 7: 'monsoon', 8: 'monsoon',
    9: 'post_monsoon', 10: 'post_monsoon', 11: 'post_monsoon'
}
data['season'] = data['month'].map(season_map)

seasonal_mean = data.groupby(['customer_id', 'season'])['consumption'].mean().reset_index()
seasonal_pivot = seasonal_mean.pivot(index='customer_id', columns='season', values='consumption')
seasonal_pivot = seasonal_pivot.fillna(seasonal_pivot.mean())

# 2. Load Factor (Consistency of usage)
data['load_factor'] = data['daily_mean'] / data.groupby('customer_id')['consumption'].transform('max')
data['load_factor'] = data['load_factor'].fillna(0)

# 3. Peer Comparison Score
global_mean = data['consumption'].mean()
data['peer_score'] = data['daily_mean'] / global_mean

# --------------------------------------------------
# FINAL FEATURE SET
# --------------------------------------------------
features = data[
    [
        'customer_id',
        'daily_mean',
        'night_ratio',
        'weekend_ratio',
        'variance',
        'sudden_drop',
        'load_factor',
        'peer_score'
    ]
].drop_duplicates()

# --------------------------------------------------
# 🔥 Consumer Risk Profiling
# --------------------------------------------------
features['risk_score'] = (
    0.3 * features['night_ratio'] +
    0.3 * features['sudden_drop'] +
    0.2 * features['variance'] +
    0.2 * features['peer_score']
)

features['risk_level'] = pd.cut(
    features['risk_score'],
    bins=[-1, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Save features
features.to_csv("data/processed/features.csv", index=False)

print("✅ STEP 5 UPGRADED SUCCESSFULLY")
print("📁 Advanced features with risk profiling generated")
