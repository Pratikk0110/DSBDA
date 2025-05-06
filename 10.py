import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows
print(titanic.head())

# Summary of the dataset
print(titanic.info())
# Survival by gender
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.show()

# Survival by class
sns.countplot(data=titanic, x='class', hue='survived')
plt.title('Survival Count by Class')
plt.show()

# Age distribution by survival
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True)
plt.title('Age Distribution by Survival')
plt.show()

# Correlation heatmap
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
