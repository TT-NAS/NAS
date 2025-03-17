"""
Módulo con funciones útiles para el manejo de los datos y modelos de segmentación de imágenes

Funciones
---------
- get_data: Obtiene los datos del dataset en forma de una lista con los nombres de archivo de los tensores
- eval_model: Evalúa un modelo en base a una lista de métricas
- train_model: Entrena un modelo UNet con los datos de un DataLoader
- plot_batch: Muestra o guarda un conjunto de imágenes y máscaras junto con sus dimensiones
- plot_results: Muestra o guarda los resultados de un modelo en un conjunto de imágenes
- empty_cache_torch: Limpia la memoria de la GPU de PyTorch
- set_current_net_binary: Establece la codificación binaria del modelo entrenandose actualmente
- save_model: Guarda un modelo en un archivo
- remove_checkpoints: Elimina los checkpoints de un modelo
"""
from .cache_manager import get_data
from .model_training import eval_model, train_model
from .data_visualization import plot_batch, plot_results
from .state import empty_cache_torch, set_current_net_binary
from .checkpoint_manager import save_model, remove_checkpoints

__all__ = [
    "get_data",
    "eval_model", "train_model",
    "plot_batch", "plot_results",
    "empty_cache_torch", "set_current_net_binary",
    "save_model", "remove_checkpoints"
]
