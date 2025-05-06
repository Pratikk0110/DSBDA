import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Bangalore_House_Data.csv")

# --- a) Handle missing values ---
print("Missing values before handling:\n", df.isnull().sum())

# Fill categorical columns with mode and numerical columns with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

print("Missing values after handling:\n", df.isnull().sum())

# --- b) Convert 'size' to numeric: '2 BHK' -> 2 ---
df['BHK'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
df['BHK'].fillna(df['BHK'].median(), inplace=True)

# --- c) Clean 'total_sqft' column ---
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            vals = x.split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        elif x.endswith('Sq. Meter') or 'Meter' in x:
            return None  # handle non-standard units if needed
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(lambda x: convert_sqft_to_num(str(x)))
df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)

# --- d) Add 'Price_Per_Sqft' column ---
df['Price_Per_Sqft'] = (df['price'] * 100000) / df['total_sqft']

# --- e) Remove outliers ---
# Remove rows with very low sqft per BHK (e.g., <300)
df = df[df['total_sqft'] / df['BHK'] >= 300]

# Remove outliers in Price_Per_Sqft using IQR
Q1 = df['Price_Per_Sqft'].quantile(0.25)
Q3 = df['Price_Per_Sqft'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Price_Per_Sqft'] >= lower_bound) & (df['Price_Per_Sqft'] <= upper_bound)]

# --- f) Linear Regression Model ---

# Select relevant features
X = df[['total_sqft', 'BHK']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Accuracy Score: {r2:.2f}")
