"""
Paquete con funciones y clases útiles para el manejo de los datos y modelos de segmentación de imágenes

Clases
------
- UNet: Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado
- TorchDataLoader: Wrapper de los DataLoaders de train, validation y test para un TorchDataset

Funciones
---------
- eval_model: Evalúa un modelo en base a una lista de métricas
- train_model: Entrena un modelo UNet con los datos de un DataLoader
- plot_batch: Muestra o guarda un conjunto de imágenes y máscaras junto con sus dimensiones
- plot_results: Muestra o guarda los resultados de un modelo en un conjunto de imágenes
- save_model: Guarda un modelo en un archivo
- remove_checkpoints: Elimina los checkpoints de un modelo
- empty_cache_torch: Limpia la memoria de la GPU de PyTorch
- set_current_net_binary: Establece la codificación binaria del modelo entrenandose actualmente

Globales
--------
- CUDA: Device al que se enviarán los tensores de PyTorch
- LOGGER: Logger del sistema

Constantes
----------
- `ROAD_BATCH_SIZE`, `CAR_BATCH_SIZE`, `COCO_BATCH_SIZE`, `CARVANA_BATCH_SIZE`: Tamaños de los
  batches para cada dataset
- `RESULTS_PATH`, `IMAGES_PATH`, `MODELS_PATH`: Rutas donde se guardarán los resultados, imágenes
  y modelos
- `ROAD_DATASET_LENGTH`, `CAR_DATASET_LENGTH`, `COCO_PEOPLE_DATASET_LENGTH`,
  `COCO_CAR_DATASET_LENGTH`, `CARVANA_DATASET_LENGTH`: Número de elementos totales de entrenamiento
  de cada dataset
- `COCO_PEOPLE_DATA_PATH`, `COCO_CAR_DATA_PATH`, `CARVANA_DATA_PATH`, `ROAD_DATA_PATH`,
  `CAR_DATA_PATH`: Rutas de los datasets
"""

from torch.amp import autocast
from torch import OutOfMemoryError, float16

from .classes import UNet, TorchDataLoader
from .functions import (
    eval_model, train_model,
    plot_batch, plot_results,
    save_model, remove_checkpoints,
    empty_cache_torch, set_current_net_binary,
    paint_real_vs_pred, paint_results
)
from .globals import CUDA, LOGGER
from .constants import (
    ROAD_BATCH_SIZE, CAR_BATCH_SIZE,
    COCO_BATCH_SIZE, CARVANA_BATCH_SIZE,

    RESULTS_PATH, IMAGES_PATH, MODELS_PATH,

    ROAD_DATASET_LENGTH, CAR_DATASET_LENGTH,
    COCO_PEOPLE_DATASET_LENGTH, COCO_CAR_DATASET_LENGTH, CARVANA_DATASET_LENGTH,

    COCO_PEOPLE_DATA_PATH, COCO_CAR_DATA_PATH,
    CARVANA_DATA_PATH, ROAD_DATA_PATH, CAR_DATA_PATH,
)

__all__ = [
    "autocast",

    "OutOfMemoryError", "float16",

    "UNet", "TorchDataLoader",

    "eval_model", "train_model",
    "plot_batch", "plot_results",
    "save_model", "remove_checkpoints",
    "empty_cache_torch", "set_current_net_binary",
    "paint_real_vs_pred", "paint_results",

    "CUDA", "LOGGER",

    "ROAD_BATCH_SIZE", "CAR_BATCH_SIZE",
    "COCO_BATCH_SIZE", "CARVANA_BATCH_SIZE",
    "RESULTS_PATH", "IMAGES_PATH", "MODELS_PATH",
    "ROAD_DATASET_LENGTH", "CAR_DATASET_LENGTH",
    "COCO_PEOPLE_DATASET_LENGTH", "COCO_CAR_DATASET_LENGTH", "CARVANA_DATASET_LENGTH",
    "COCO_PEOPLE_DATA_PATH", "COCO_CAR_DATA_PATH",
    "CARVANA_DATA_PATH", "ROAD_DATA_PATH", "CAR_DATA_PATH",
]
