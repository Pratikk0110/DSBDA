import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
df = pd.read_csv("Academic_Performance_Dataset.csv")

# --- (a) MISSING VALUES & INCONSISTENCIES ---

# Check for missing values
missing_summary = df.isnull().sum()
print("Missing Values per Column:\n", missing_summary)

# Check for inconsistent values (e.g., negative marks, typos in categories)
print("\nSummary Statistics:\n", df.describe(include='all'))

# Handling missing values:
# - For numerical variables: fill with mean/median
# - For categorical variables: fill with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# --- (b) OUTLIERS IN NUMERIC VARIABLES ---

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Use boxplot to visualize outliers
for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Handling outliers using IQR method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                       np.where(df[col] > upper_bound, upper_bound, df[col]))

# --- (c) TRANSFORM CATEGORICAL VARIABLES TO NUMERIC ---

# Identify categorical variables
cat_cols = df.select_dtypes(include=['object']).columns

# Label encoding or one-hot encoding depending on cardinality
for col in cat_cols:
    unique_vals = df[col].nunique()
    if unique_vals <= 5:
        # Few unique categories: use one-hot encoding
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:
        # Many unique categories: use label encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

print("\nTransformed Dataset:\n", df.head())
