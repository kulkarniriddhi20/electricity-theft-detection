import pandas as pd

# Load dataset
data = pd.read_csv("data/raw/smart_meter.csv")

print("Dataset loaded successfully")
print("Shape of dataset:", data.shape)

print("\nColumn names:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())
