import pandas as pd
from codec import Chromosome
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv("./results/results_with_params.csv")
# Ordenar primero por vaL_iou y después por número de parámetros
df = df.sort_values(by=['val_iou', 'num_parameters'], ascending=[False, True])

# Dividir en grupos con rangos de 0.1 en val_iou
df['group'] = pd.cut(df['val_iou'], bins=10)

# Tomar dos de cada grupo
df_filtered = df.groupby('group').head(2).reset_index(drop=True)
df_filtered = df_filtered[["binary codification", 'val_iou', 'num_parameters']]
df_filtered.to_csv("./results/results_filtered.csv", index=False)
