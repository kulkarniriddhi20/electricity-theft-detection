import pandas as pd
import numpy as np

# Load clean data
data = pd.read_csv("data/processed/clean_data.csv")

# Add label column (default = normal)
data['label'] = 0

# Select random customers for theft
np.random.seed(42)
theft_customers = np.random.choice(
    data['customer_id'].unique(),
    size=int(0.1 * data['customer_id'].nunique()),  # 10% theft users
    replace=False
)

# Apply theft behavior
for cust in theft_customers:
    idx = data[data['customer_id'] == cust].index
    
    # Reduce consumption artificially (theft simulation)
    data.loc[idx, 'consumption'] = data.loc[idx, 'consumption'] * np.random.uniform(0.2, 0.5)
    
    # Mark as theft
    data.loc[idx, 'label'] = 1

# Save final dataset
data.to_csv("data/processed/final_data.csv", index=False)

print("✅ Theft simulation and labeling completed")
print(data['label'].value_counts())
