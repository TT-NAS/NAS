import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

IMAGES_PATH = "./results_carvana/imgs/"
PATIENCE = 5
NUM_MODELS = 30


def detectar_epoca_maxima(loss):
    smooth = savgol_filter(loss, window_length=5, polyorder=2)
    best_epoch = np.argmin(smooth)
    overfit_start = None

    for i in range(best_epoch + 1, len(smooth) - PATIENCE):
        if all(smooth[i + j] > smooth[i + j - 1] for j in range(1, PATIENCE + 1)):
            overfit_start = i
            break

    if overfit_start:
        return overfit_start
    else:
        return len(loss) - 1


dirs = [d for d in os.listdir(IMAGES_PATH)
        if os.path.isdir(os.path.join(IMAGES_PATH, d))]
dirs = [d for d in dirs
        if any(d.startswith(f"s{i}-") for i in range(NUM_MODELS))]
max_epochs = []

for dir in dirs:
    ruta_carpeta = os.path.join(IMAGES_PATH, dir)
    csv = [f for f in os.listdir(
        ruta_carpeta) if f.startswith("learning curves carvana") and f.endswith(".csv")]

    if not csv:
        print(f"No se encontró CSV para {dir}")
        continue

    path_csv = os.path.join(ruta_carpeta, csv[0])
    df = pd.read_csv(path_csv)
    loss = df["val_loss"].values
    # loss = df["train_loss"].values

    max_epochs.append(detectar_epoca_maxima(loss))

mean_epoch = int(np.mean(max_epochs))
median_epoch = int(np.median(max_epochs))
p90_epoch = int(np.percentile(max_epochs, 90))

print("Época máxima sugerida (media):", mean_epoch)
print("Época máxima sugerida (mediana):", median_epoch)
print("Época máxima sugerida (percentil 90):", p90_epoch)
