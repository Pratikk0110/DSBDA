import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load NBA dataset
nba_df = pd.read_csv('NBA.csv')

# Load Dirty data
dirty_df = pd.read_csv('Dirtydata.csv')

# NBA Dataset
print("NBA Dataset Overview:\n")
print(nba_df.head())
print(nba_df.info())
print("\nMissing Values:\n", nba_df.isnull().sum())
print("Shape of Data:", nba_df.shape)

# Dirty Dataset
print("\nDirty Data Overview:\n")
print(dirty_df.head())
print(dirty_df.info())
print("\nMissing Values:\n", dirty_df.isnull().sum())
print("Shape of Data:", dirty_df.shape)

# NBA
print("\nNBA Variable Types:\n", nba_df.dtypes.value_counts())
print("\nNBA Column Data Types:\n", nba_df.dtypes)

# Dirty Data
print("\nDirty Data Variable Types:\n", dirty_df.dtypes.value_counts())
print("\nDirty Data Column Data Types:\n", dirty_df.dtypes)

# NBA Dataset
if 'Salary' in nba_df.columns:
    nba_df['Salary'] = nba_df['Salary'].replace('[\$,]', '', regex=True).astype(float)

# Dirty Dataset
if 'Date' in dirty_df.columns:
    dirty_df['Date'] = pd.to_datetime(dirty_df['Date'], errors='coerce')

if 'Age' in dirty_df.columns:
    dirty_df['Age'] = pd.to_numeric(dirty_df['Age'], errors='coerce')

print("\nAfter Type Conversion:")
print(dirty_df.dtypes)

# NBA Dataset - One-hot encoding for 'Team' or 'Position' if they exist
if 'Team' in nba_df.columns:
    nba_df = pd.get_dummies(nba_df, columns=['Team'], drop_first=True)

if 'Position' in nba_df.columns:
    nba_df = pd.get_dummies(nba_df, columns=['Position'], drop_first=True)

# Dirty Data – Label encode 'Gender'
if 'Gender' in dirty_df.columns:
    dirty_df['Gender_encoded'] = dirty_df['Gender'].map({'Male': 1, 'Female': 0})

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Normalize NBA numeric columns (e.g., Salary, Age, etc.)
nba_numeric_cols = nba_df.select_dtypes(include=[np.number]).columns
nba_df[nba_numeric_cols] = scaler.fit_transform(nba_df[nba_numeric_cols])

print("\nNormalized NBA Data:\n", nba_df.head())
