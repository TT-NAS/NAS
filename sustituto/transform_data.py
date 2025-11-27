"""
Módulo para transformar el dataset de redes neuronales entrenadas para trabajar con el modelo sustituto
"""
import pandas as pd
import os
import sys
from tqdm import tqdm

# Añadir la ruta del directorio src al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import codec

def get_real_and_binary_chromosoma(filename: str = "results.csv", out_filename: str = "results_transformed.csv", max_layers: int = 4, max_conv_per_layer: int = 2) -> None:
  """
  Transforma el dataset de redes neuronales entrenadas, agrega las columnas de codificación real y binaria, crea un nuevo archivo csv con el dataset transformado

  Parameters
  ----------
  filename : str
    Nombre del archivo csv que contiene el dataset de redes neuronales entrenadas
  max_layers : int
    Número máximo de capas que puede tener una red neuronal
  max_conv_per_layer : int
    Número máximo de capas convolucionales que puede tener una red neuronal

  Returns
  -------
  None
  """
  df = pd.read_csv(filename)
  # Crea dos nuevas columnas en el dataframe
  df["real_codification"] = None
  # Itera sobre las filas
  for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Transforming dataset"):
    chromosome = row["binary codification"]
    try:
      c = codec.Chromosome(chromosome = chromosome)
      # Obtiene la arquitectura en codificación real
      c_real = c.get_real()
      # crea una nueva columna en el dataframe
      df.at[index, "real_codification"] = c_real
    except Exception as e:
      print(f"Error processing row {index}: {e}")
  # Guarda el dataframe
  df.to_csv(out_filename, index=False)

if __name__ == "__main__":
  get_real_and_binary_chromosoma(filename="./results_max_epoch.csv", out_filename="results_transformed_2376.csv")
