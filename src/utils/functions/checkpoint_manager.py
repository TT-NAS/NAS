"""
Módulo con funciones para el manejo de los checkpoints y el guardado de modelos

Funciones
---------
- save_model: Guarda un modelo en un archivo
- remove_checkpoint: Elimina los checkpoints de un modelo
- set_checkpoint: Guarda un checkpoint del modelo actual
- load_checkpoints: Carga un checkpoint del modelo
"""
import os
import pickle
from typing import Optional

import torch

import utils.globals as g
from ..classes import UNet
from ..constants import CHECKPOINT_PATH, MODELS_PATH


def save_model(model: UNet, name: str, path: str = MODELS_PATH):
    """
    Guarda un modelo en un archivo

    Parameters
    ----------
    model : UNet
        Modelo a guardar
    name : str
        Nombre del archivo
    path : str, optional
        Ruta donde se guardará el modelo, by default `MODELS_PATH`
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, name))

    print(f"-> Modelo guardado en {os.path.join(path, name)}")


def remove_checkpoints(net_binary: str = Optional[None], path: str = CHECKPOINT_PATH):
    """
    Elimina los checkpoints de un modelo

    Parameters
    ----------
    net_binary : Optional[str], optional
        Arquitectura del modelo en codificación binaria (`zip=True`), by default None
    path : str, optional
        Ruta donde se buscarán los checkpoints, by default `CHECKPOINT`
    """
    if net_binary is not None:
        net_binary = g.CURRENT_NET_BINARY

    path = os.path.join(path, net_binary)

    if not os.path.exists(path):
        return

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))

    os.rmdir(path)


def set_checkpoint(model_state: dict[str, any], metrics_results: dict[str, list[float]],
                   current_epoch: int, path: str = CHECKPOINT_PATH):
    """
    Guarda un checkpoint del modelo actual

    Parameters
    ----------
    model : UNet
        Modelo a guardar
    current_epoch : int
        Última época completada
    path : str, optional
        Ruta donde se guardará el checkpoint, by default `CHECKPOINT_PATH`
    """
    remove_checkpoints(path=path)

    path = os.path.join(path, g.CURRENT_NET_BINARY)

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    model_name = f"{current_epoch}.pt"
    torch.save(model_state, os.path.join(path, model_name))

    with open(os.path.join(path, "metrics_results.pickle"), "wb") as file:
        pickle.dump(metrics_results, file)


def load_checkpoint(model: UNet, path: str = CHECKPOINT_PATH) -> tuple[UNet,
                                                                       Optional[dict[str,
                                                                                     list[float]]],
                                                                       int]:
    """
    Carga un checkpoint del modelo

    Parameters
    ----------
    model : UNet
        Modelo a cargar
    path : str, optional
        Ruta donde se buscará el checkpoint, by default `CHECKPOINT_PATH`

    Returns
    -------
    tuple
        (Modelo cargado, resultados de las métricas del checkpoint,
        época inicial para continuar el entrenamiento)
    """
    path = os.path.join(path, g.CURRENT_NET_BINARY)

    if not os.path.exists(path):
        return model, None, 0

    print("Se encontró un checkpoint para el modelo actual")

    checkpoint_data = os.listdir(path)

    if len(checkpoint_data) < 2:
        print("Los datos del checkpoint no están completos. No se cargará el checkpoint.")

        return model, None, 0

    print("Cargando los datos del último checkpoint...")

    metrics_file = checkpoint_data[-1]
    checkpoints_files = sorted(checkpoint_data[:-1])
    checkpoint = checkpoints_files[-1]

    model.load_state_dict(torch.load(os.path.join(path, checkpoint)))

    with open(os.path.join(path, metrics_file), "rb") as file:
        metrics_results = pickle.load(file)

    return model, metrics_results, int(checkpoint.split(".")[0]) + 1
