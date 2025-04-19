""" 
Módulo para transformar el dataset de redes neuronales entrenadas para trabajar con el modelo sustituto
"""
import pandas as pd
import os
import sys

# Añadir la ruta del directorio src al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import Codec

def get_real_and_binary_chromosoma(filename: str = "results.csv", max_layers: int = 3, max_conv_per_layer: int = 2) -> None:
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
  df["binary_codification"] = None
  # Itera sobre las filas
  for index, row in df.iterrows():
    chromosome = row["id"].split("_", 1)[-1] # Obtiene el cromosoma sin su id
    c = Codec.Chromosome(max_layers = max_layers, max_conv_per_layer = max_conv_per_layer, chromosome = chromosome)
    # Obtiene la arquitectura en codificación real
    c_real = c.get_real()
    c_binary = c.get_binary()
    # crea una nueva columna en el dataframe
    df.at[index, "real_codification"] = c_real
    df.at[index, "binary_codification"] = c_binary
  # Guarda el dataframe
  df.to_csv(f"results_transformed.csv", index=False)
  
if __name__ == "__main__":
  get_real_and_binary_chromosoma()