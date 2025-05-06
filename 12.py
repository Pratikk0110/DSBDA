import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = sns.load_dataset('iris')

# Display the first few rows
print(iris.head())

# Summary
print(iris.info())
# Determine data types
print(iris.dtypes)
iris.hist(figsize=(10, 8), bins=20, edgecolor='black')
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()
# One boxplot per numeric column
plt.figure(figsize=(12, 8))
for i, column in enumerate(iris.select_dtypes(include='number').columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=iris[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
# Detect outliers using IQR method
for col in iris.select_dtypes(include='number').columns:
    Q1 = iris[col].quantile(0.25)
    Q3 = iris[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = iris[(iris[col] < lower_bound) | (iris[col] > upper_bound)]
    print(f"\nOutliers in {col}: {len(outliers)}")
    if not outliers.empty:
        print(outliers[[col, 'species']])
