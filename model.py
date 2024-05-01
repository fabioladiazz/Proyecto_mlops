import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

"""# Data Exploring"""

data = pd.read_csv('house.csv')
data.head()

missing_values = data.isna().sum()

print("Valores faltantes por columna:")
print(missing_values)

data.drop(['id', 'date'], axis=1, inplace=True)

"""# Exploratory Data Analysis"""

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='long', y='lat', data=data, color='blue', alpha=0.5)
plt.title('Distribución espacial de las casas')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

"""## Regresion Múltiple"""

data = pd.read_csv('house.csv')

X = data.drop(['price', 'id', 'date'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

y_pred_linear = model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)

print("MSE para Regresión Lineal Múltiple:", mse_linear)
print("R^2 en el conjunto de prueba:", test_score)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
