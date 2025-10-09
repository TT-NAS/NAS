import os
import pandas as pd
import torch
from tqdm import tqdm

# --- Ajustes ---
CSV_IN = "results/results_with_params.csv"
CSV_OUT = "results/selected_networks.csv"
INPUT_RES = (3, 256, 256)  # (C, H, W) -> ajusta si tu U-Net usa otra resolución
BATCH_SIZE = 1             # batch dummy para el conteo
BINS = 10                  # número de grupos por val_iou
K_PER_BIN = 10             # cuántos modelos seleccionar por bin (los de menos FLOPs)

# --- Imports del proyecto ---
from codec import Chromosome

# --- FLOPs helper (THOP) ---
def compute_flops_thop(model, input_shape):
    """
    Devuelve FLOPs como número (no string). Usa thop.profile.
    input_shape: (C, H, W)
    """
    try:
        from thop import profile
    except ImportError as e:
        raise RuntimeError(
            "No se encontró 'thop'. Instálalo con: pip install thop"
        ) from e

    model.eval()
    model_cpu = model.cpu()
    C, H, W = input_shape
    dummy = torch.randn(BATCH_SIZE, C, H, W, dtype=torch.float32)

    with torch.no_grad():
        flops, params = profile(model_cpu, inputs=(dummy,), verbose=False)

    # thop reporta 'flops' como número de operaciones (aprox. MACs*2 para algunos casos)
    # Aquí devolvemos el valor bruto de FLOPs (operaciones).
    return float(flops)

# --- Carga de datos ---
df = pd.read_csv(CSV_IN)

# Verificaciones mínimas
required_cols = {"binary codification", "val_iou"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

# --- Cálculo de FLOPs ---
flops_list = []
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculando FLOPs"):
    c = Chromosome(chromosome=row["binary codification"])
    model = c.get_unet()
    flops_value = compute_flops_thop(model, INPUT_RES)
    flops_list.append(flops_value)

df["flops"] = flops_list

# --- Binning por val_iou (deciles) ---
# Usamos qcut para crear 10 grupos aproximadamente balanceados por cuantiles.
# duplicates='drop' evita errores si hay muchos empates.
df["val_iou_bin"] = pd.qcut(df["val_iou"], q=BINS, labels=False, duplicates="drop")

# En caso extremo de pocos valores únicos, puede que se creen < BINS bins.
bins_creados = int(df["val_iou_bin"].nunique())
if bins_creados < BINS:
    print(f"Advertencia: solo se pudieron crear {bins_creados} bins por empates en 'val_iou'.")

# --- Selección: 10 con menos FLOPs por bin ---
def pick_k_least_flops(group, k=K_PER_BIN):
    # Ordena por FLOPs y toma los k primeros
    return group.sort_values("flops", ascending=True).head(k)

df_selected = (
    df.dropna(subset=["val_iou_bin"])  # por si qcut dejó NAs
      .groupby("val_iou_bin", group_keys=False)
      .apply(pick_k_least_flops, k=K_PER_BIN)
      .reset_index(drop=True)
)

# --- Guardar resultado ---
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
df_selected.to_csv(CSV_OUT, index=False)

print(f"Listo. Se guardó la selección en: {CSV_OUT}")
print(df_selected.head())