"""
Módulo con funciones y clases útiles para trabajar con PyTorch:

Clases
------
- CarvanaDataset: Clase para cargar el dataset de Carvana
- RoadDataset: Clase para cargar el dataset de carreteras
- CarDataset: Clase para cargar el dataset de carros
- TorchDataLoader: Wrapper de los DataLoaders de train, validation y test para un dataset
    - No es necesario instanciar esta clase, se puede usar directamente un DataLoader de PyTorch,
      pero esta clase facilita el proceso de cargar los datos
- UNet: Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado
- Synflow: Clase para calcular el Synflow de un modelo UNet

Funciones
---------
- plot_batch: Muestra o guarda un conjunto de imágenes y máscaras junto con sus dimensiones
- plot_results: Muestra o guarda los resultados de un modelo en un conjunto de imágenes
- dice_loss: Calcula la pérdida de Dice
- dice_crossentropy_loss: Calcula la pérdida de Dice-Crossentropy
- iou_loss: Calcula la pérdida de IoU
- accuracy: Calcula la precisión de la máscara predicha
- eval_model: Evalúa un modelo en base a una lista de métricas
- train_model: Entrena un modelo UNet
- save_model: Guarda un modelo en un archivo
- gradient_scorer_pytorch: No jala
"""

from .classes import (
    CarvanaDataset, RoadDataset, CarDataset,
    TorchDataLoader,
    UNet,
    Synflow
)
from .functions import (
    plot_batch, plot_results,
    dice_loss, dice_crossentropy_loss, iou_loss, accuracy_loss,
    eval_model, train_model, save_model,
    gradient_scorer_pytorch
)
from .constants import (
    CARVANA_DATASET_LENGTH, ROAD_DATASET_LENGTH, CAR_DATASET_LENGTH,
    CARVANA_BATCH_SIZE, ROAD_BATCH_SIZE, CAR_BATCH_SIZE, SHOW_SIZE,
    CARVANA_DATA_PATH, ROAD_DATA_PATH, CAR_DATA_PATH,
    MODELS_PATH, IMAGES_PATH,
    WIDTH, HEIGHT, CHANNELS,
    TRANSFORM,
    CUDA,
    LOGGER
)

__all__ = [
    "CarvanaDataset", "RoadDataset", "CarDataset",
    "TorchDataLoader",
    "UNet",
    "Synflow",

    "plot_batch", "plot_results",
    "dice_loss", "dice_crossentropy_loss", "iou_loss", "accuracy_loss",
    "eval_model", "train_model", "save_model",
    "gradient_scorer_pytorch",

    "CARVANA_DATASET_LENGTH", "ROAD_DATASET_LENGTH", "CAR_DATASET_LENGTH",
    "CARVANA_BATCH_SIZE", "ROAD_BATCH_SIZE", "CAR_BATCH_SIZE", "SHOW_SIZE",
    "WIDTH", "HEIGHT", "CHANNELS",
    "CARVANA_DATA_PATH", "ROAD_DATA_PATH", "CAR_DATA_PATH",
    "MODELS_PATH", "IMAGES_PATH",
    "TRANSFORM",
    "CUDA",
    "LOGGER"
]
