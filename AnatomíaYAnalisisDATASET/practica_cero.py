import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Generación de datos sintéticos
np.random.seed(42)
tiempo = pd.date_range(start='2026-04-15', periods=500, freq='min')

valores = (
    np.linspace(10, 50, 500) +
    np.sin(np.linspace(0, 20, 500)) * 5 +
    np.random.normal(0, 2, 500)
)

df = pd.DataFrame({
    'tiempo': tiempo,
    'valor': valores
})

# 2. Inyección de anomalías intencionales

# Outliers
df.loc[50, 'valor'] = 100
df.loc[120, 'valor'] = -20

# Valores nulos
df.loc[200:210, 'valor'] = np.nan

# Cambio abrupto de nivel
df.loc[300:350, 'valor'] = df.loc[300:350, 'valor'] + 15

# 3. Inspección general
print("Primeras filas:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df['valor'].describe())

print("\nValores nulos:")
print(df.isnull().sum())

print("\nMáximo:", df['valor'].max())
print("Mínimo:", df['valor'].min())
print("Media:", df['valor'].mean())
print("Desviación estándar:", df['valor'].std())

# 4. Gráfica de la serie temporal
plt.figure(figsize=(12, 5))
plt.plot(df['tiempo'], df['valor'])
plt.title('Serie temporal con anomalías')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Histograma
plt.figure(figsize=(8, 5))
df['valor'].hist(bins=30)
plt.title('Distribución de la señal')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# 6. Boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df['valor'].dropna())
plt.title('Boxplot de la señal')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()