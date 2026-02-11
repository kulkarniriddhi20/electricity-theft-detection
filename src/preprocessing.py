import pandas as pd

# Load raw data
data = pd.read_csv("data/raw/smart_meter.csv")

# Rename columns
data.rename(columns={
    'LCLid': 'customer_id',
    'tstp': 'datetime',
    'energy(kWh/hh)': 'consumption'
}, inplace=True)

# Convert datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Convert consumption to numeric (IMPORTANT FIX)
data['consumption'] = pd.to_numeric(data['consumption'], errors='coerce')

# Sort by customer and time
data = data.sort_values(by=['customer_id', 'datetime'])

# Forward fill missing values
data['consumption'] = data['consumption'].ffill()

# Remove invalid values
data = data[data['consumption'] >= 0]

# Reset index
data.reset_index(drop=True, inplace=True)

# Save cleaned data
data.to_csv("data/processed/clean_data.csv", index=False)

print("✅ Data preprocessing completed successfully")
print(data.head())
