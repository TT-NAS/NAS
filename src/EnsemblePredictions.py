import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("results/selected_networks_scored.csv")
# Features y target
X = df[["gradnorm", "synflow", "RN", "NTK"]]
y = df["val_iou"]

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definir y entrenar el modelo
rf = RandomForestRegressor(
    n_estimators=10,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predicciones en test
y_pred = rf.predict(X_test)

# MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"MSE en test set: {mse:.4f}")
# Correlación de Pearson
pearson_corr, pearson_pval = pearsonr(y_test, y_pred)

# Correlación de Spearman
spearman_corr, spearman_pval = spearmanr(y_test, y_pred)

print("Resultados en test set:")
print(f"Pearson: {pearson_corr:.4f} (p={pearson_pval:.4e})")
print(f"Spearman: {spearman_corr:.4f} (p={spearman_pval:.4e})")

# (Opcional) Importancia de features
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nImportancia de features:")
print(importances.sort_values(ascending=False))

# Grafica real vs predicho
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title("Predicciones vs Valores Reales")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()