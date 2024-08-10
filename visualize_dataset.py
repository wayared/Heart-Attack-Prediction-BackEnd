import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Cargar el dataset
data = datasets.load_breast_cancer()

# Crear un DataFrame de pandas
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

# Mostrar las primeras filas del DataFrame
print(df.head())

# Histograma
plt.figure(figsize=(8, 6))
plt.hist(df['mean radius'], bins=30, edgecolor='k')
plt.xlabel('Mean Radius')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Radius')
plt.show()

# Gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(df['mean radius'], df['mean texture'], c=df['target'], cmap='viridis', alpha=0.7)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Scatter plot of Mean Radius vs Mean Texture')
plt.colorbar(label='Target')
plt.show()

# Matriz de correlación
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df, hue='target', palette='viridis', markers=['o', 's'], diag_kind='kde')
plt.show()
