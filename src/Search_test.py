from search_algorithms import SurrogateModel, DiferentialEvolution
import matplotlib.pyplot as plt
import pandas as pd

# Cargar resultados de la UNet
df =pd.read_csv("./results/base_architectures/k_folds/unet.csv")
# Media y stdv ("val_iou")
mean = df["val_iou"].mean()
stdv = df["val_iou"].std()

print(f"UNet val_iou mean: {mean}, stdv: {stdv}")

gold_standard = mean
tolerance = 2 * stdv

surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
de = DiferentialEvolution()
de.start()

plt.plot(de.lower, label="Lower")
plt.plot(de.upper, label="Upper")
plt.plot(de.mean,  label="Mean")

plt.axhline(gold_standard, linestyle="--", linewidth=1, label=f"Gold standard = {gold_standard}")
plt.axhspan(gold_standard - tolerance, gold_standard + tolerance, alpha=0.1, label=f"±{tolerance} tol")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Differential Evolution Fitness")
plt.legend()
plt.show()