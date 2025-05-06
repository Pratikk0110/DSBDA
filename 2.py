import pandas as pd
import numpy as np
import math

# Load the dataset
df = pd.read_csv("Age-Income-Dataset.csv")

# ----- 1. SUMMARY STATISTICS (WITH LIBRARY FUNCTIONS) -----
print("Summary using library functions:")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    print(f"\nStatistics for {col}:")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Min: {df[col].min():.2f}")
    print(f"Max: {df[col].max():.2f}")
    print(f"Standard Deviation: {df[col].std():.2f}")

# ----- 1. SUMMARY STATISTICS (WITHOUT LIBRARY FUNCTIONS) -----
print("\nSummary without using library functions:")

def calculate_manual_stats(data):
    data_sorted = sorted(data)
    n = len(data)
    mean = sum(data) / n
    median = data_sorted[n // 2] if n % 2 != 0 else (data_sorted[n // 2 - 1] + data_sorted[n // 2]) / 2
    minimum = data_sorted[0]
    maximum = data_sorted[-1]
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_dev = math.sqrt(variance)
    return mean, median, minimum, maximum, std_dev

for col in numeric_cols:
    data_list = df[col].dropna().tolist()
    mean, median, min_val, max_val, std_dev = calculate_manual_stats(data_list)
    print(f"\nManual stats for {col}:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Min: {min_val:.2f}")
    print(f"Max: {max_val:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")

# ----- 2. SUMMARY OF INCOME GROUPED BY AGE GROUPS -----
# Create Age Groups (e.g., <20, 20–30, 31–40, etc.)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 100],
                         labels=['<20', '21-30', '31-40', '41-50', '51-60', '60+'])

# Group by Age_Group and calculate summary for Income
print("\nIncome Summary by Age Group:")
grouped = df.groupby('Age_Group')['Income'].agg(['mean', 'median', 'min', 'max', 'std'])
print(grouped)

# ----- 3. NUMERIC VALUES FOR CATEGORICAL VARIABLES -----
categorical_cols = df.select_dtypes(include=['object']).columns
category_mappings = {}

print("\nCategorical variable mappings:")
for col in categorical_cols:
    unique_vals = df[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    category_mappings[col] = mapping
    print(f"{col}: {mapping}")

# Create a list of numeric values for each response
encoded_lists = {}
for col in categorical_cols:
    encoded_lists[col] = [category_mappings[col][val] for val in df[col]]

print("\nEncoded lists for categorical variables:")
for col, encoded in encoded_lists.items():
    print(f"{col}: {encoded[:10]}...")  # show first 10 values
