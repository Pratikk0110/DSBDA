import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column to the dataframe
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display first few rows
print(df.head())
print(df.dtypes)
# Set up a figure with subplots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot histogram for each feature
sns.histplot(df['sepal length (cm)'], kde=True, ax=axes[0, 0], color='skyblue').set_title('Sepal Length Distribution')
sns.histplot(df['sepal width (cm)'], kde=True, ax=axes[0, 1], color='orange').set_title('Sepal Width Distribution')
sns.histplot(df['petal length (cm)'], kde=True, ax=axes[1, 0], color='green').set_title('Petal Length Distribution')
sns.histplot(df['petal width (cm)'], kde=True, ax=axes[1, 1], color='red').set_title('Petal Width Distribution')

# Adjust layout for better viewing
plt.tight_layout()
plt.show()
# Set up a figure with subplots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Create boxplots for each feature
sns.boxplot(x=df['species'], y=df['sepal length (cm)'], ax=axes[0, 0], palette="Set2").set_title('Sepal Length Boxplot')
sns.boxplot(x=df['species'], y=df['sepal width (cm)'], ax=axes[0, 1], palette="Set2").set_title('Sepal Width Boxplot')
sns.boxplot(x=df['species'], y=df['petal length (cm)'], ax=axes[1, 0], palette="Set2").set_title('Petal Length Boxplot')
sns.boxplot(x=df['species'], y=df['petal width (cm)'], ax=axes[1, 1], palette="Set2").set_title('Petal Width Boxplot')

# Adjust layout for better viewing
plt.tight_layout()
plt.show()
# Function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Example: Check for outliers in 'sepal length (cm)'
outliers_sepal_length = detect_outliers(df, 'sepal length (cm)')
print(outliers_sepal_length)
