import json 
import os

import numpy as np
from scipy.stats import ttest_rel, t

path = "./output/folds"
base_model = "k_fold_results_0.json"

results = {}
for file in os.listdir(path):
    path_file = os.path.join(path, file)
    with open(path_file, "r") as f:
        data = json.load(f)
        
    folds = data["folds"]
    val_ious = [f["validation_iou"] for f in folds]
    results[file] = np.array(val_ious)

fold_count = len(next(iter(results.values())))
base_model_ious = results[base_model]

# Estadísticas del modelo base
base_mean = np.mean(base_model_ious)
base_std = np.std(base_model_ious, ddof=1)

print(f"=== Modelo base: {base_model} ===")
print(f"Media IoU: {base_mean:.4f}")
print(f"Std: {base_std:.4f}")
print()

# Comparar todos los modelos contra el base
for name, ious in results.items():
    if name == base_model:
        continue

    mean = np.mean(ious)
    std = np.std(ious, ddof=1)

    t_crit = t.ppf(0.975, df=fold_count - 1)
    ci_half = t_crit * std / np.sqrt(fold_count)
    ci_low, ci_high = mean - ci_half, mean + ci_half

    # Prueba t apareada contra el modelo base
    t_stat, p_val = ttest_rel(ious, base_model_ious)

    print(f"=== Modelo: {name} ===")
    print(f"Media IoU: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"IC 95%: ({ci_low:.4f}, {ci_high:.4f})")
    print(f"t-stat: {t_stat:.4f} | p-value: {p_val:.4f}")
    if p_val > 0.05:
        print("- No hay diferencia significativa (p > 0.05)")
    else:
        print("- Hay diferencia significativa (p <= 0.05)")
    print()
