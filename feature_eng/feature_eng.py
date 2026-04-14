import pandas as pd
import numpy as np

df = pd.read_csv("../data/processed/preprocessed_master.csv")
df = pd.DataFrame(df)

# 1. Create log_target
df = df[df['total_recovery_cost'] >= 0]
df['log_target'] = np.log1p(df['total_recovery_cost'])

# 2. Drop rows where target is null or zero
df = df.dropna(subset=['log_target'])

# 3. Group rare incident_type
counts = df['incident_type'].value_counts()
rare = counts[counts <= 10].index
df['incident_type'] = df['incident_type'].replace(rare, 'Other')
df = pd.get_dummies(df, columns=['incident_type'], dtype=int)

# 4. Fix bool columns
bool_cols = ['ih_declared', 'ia_declared', 'pa_declared', 'hm_declared', 'tribal_request']
df[bool_cols] = df[bool_cols].astype(int)

print(df.shape)
print(df.dtypes)

df.to_csv("../data/processed/final_master.csv", index=False)