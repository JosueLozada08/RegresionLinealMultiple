import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Cargar datos desde Input ===
df = pd.read_csv("Input/Walmart.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# === 2. Variables independientes y dependiente ===
X = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
y = df['Weekly_Sales']

# === 3. División en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Entrenamiento del modelo ===
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 5. Evaluación ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === 6. Mostrar resultados en consola ===
print("\n===== RESULTADOS DEL MODELO =====")
print(f"Intercepto: {model.intercept_:.2f}")
print("\nCoeficientes:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.2f}")
print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# === 7. Crear carpeta Output si no existe ===
os.makedirs("Output", exist_ok=True)

# === 8. Histograma de Weekly Sales ===
plt.figure(figsize=(10, 5))
sns.histplot(df['Weekly_Sales'], bins=30, kde=True)
plt.title("Distribución de Weekly Sales")
plt.xlabel("Weekly Sales")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("Output/histograma_weekly_sales.png")
plt.show()

# === 9. Matriz de correlación ===
plt.figure(figsize=(10, 8))
corr = df[['Weekly_Sales', 'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig("Output/correlacion.png")
plt.show()

# === 10. Gráfico: Predicciones vs Reales ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Ventas reales")
plt.ylabel("Ventas predichas")
plt.title("Predicciones vs Valores Reales")
plt.tight_layout()
plt.savefig("Output/predicciones_vs_reales.png")
plt.show()

# === 11. Guardar resumen en archivo de texto ===
with open("Output/resumen_resultados.txt", "w", encoding="utf-8") as f:
    f.write(f"Intercepto: {model.intercept_:.2f}\n")
    f.write("Coeficientes:\n")
    for name, coef in zip(X.columns, model.coef_):
        f.write(f"  {name}: {coef:.2f}\n")
    f.write(f"\nMSE: {mse:.2f}\n")
    f.write(f"R²: {r2:.4f}\n")
