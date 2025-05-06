import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Check the column names to ensure consistency
print("Dataset Columns:", df.columns)

# List of species
species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# --- BASIC STATISTICAL DETAILS BY SPECIES ---
for species in species_list:
    print(f"\n--- Statistics for {species} ---")
    species_data = df[df['species'] == species]
    print(species_data.describe(percentiles=[.25, .5, .75]))

# --- MEASURES OF VARIABILITY ---
def variability_measures(data):
    variability = {}
    for col in data.columns:
        if data[col].dtype != 'object':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            variability[col] = {
                'Range': data[col].max() - data[col].min(),
                'IQR': Q3 - Q1,
                'Variance': data[col].var(),
                'Std Dev': data[col].std()
            }
    return pd.DataFrame(variability)

print("\n--- Measures of Variability by Species ---")
for species in species_list:
    print(f"\n{species} Variability:")
    species_data = df[df['species'] == species].drop(columns='species')
    print(variability_measures(species_data))

# --- CORRELATION MATRIX AND HEATMAP ---
numeric_df = df.drop(columns='species')
correlation_matrix = numeric_df.corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix)

# Visualization using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Iris Dataset Features")
plt.show()
