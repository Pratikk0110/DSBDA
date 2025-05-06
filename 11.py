import seaborn as sns
import matplotlib.pyplot as plt

# Box plot: Age distribution by gender and survival
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='sex', y='age', hue='survived', palette='Set2')
plt.title('Box Plot of Age by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()
