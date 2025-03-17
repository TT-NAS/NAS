"""
Módulo con funciones para la visualización de datos

Funciones
---------
- plot_batch: Muestra o guarda un conjunto de imágenes y máscaras junto con sus dimensiones
- plot_results: Muestra o guarda los resultados de un modelo en un conjunto de imágenes
"""
import os
import math
from typing import Union

import torch
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ..classes import UNet
from ..globals import CUDA
from ..constants import SHOW_SIZE, IMAGES_PATH


def plot_batch(imgs: Tensor, masks: Tensor, save: bool = False, show_size: int = SHOW_SIZE,
               name: str = "batch.png", path: str = IMAGES_PATH):
    """
    Muestra un conjunto de imágenes y máscaras junto con sus dimensiones

    Parameters
    ----------
    imgs : Tensor
        Imágenes
    masks : Tensor
        Máscaras
    save : bool, optional
        Si se guardan las imágenes o se muestran, by default `False`
    show_size : int, optional
        Número de imágenes a mostrar, by default `SHOW_SIZE`
    name : str, optional
        Nombre del archivo a guardar, by default `"batch.png"`
    path : str, optional
        Ruta donde se guardarán las imágenes, by default `IMAGES_PATH`
    """
    imgs = imgs.clone().float()
    masks = masks.clone().float()

    # Mostrar un batch de imágenes y máscaras
    cols = math.ceil(math.sqrt(show_size))
    rows = math.ceil(show_size / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(show_size):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i, ...].permute(1, 2, 0).numpy()
        mask = masks[i, ...].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.axis("Off")

    plt.tight_layout()

    if save:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        plt.savefig(os.path.join(path, name))

        print(
            f"-> Gráfico de prueba con el batch guardado en {os.path.join(path, name)}"
        )
    else:
        plt.show()

    plt.close()


def plot_results(model: UNet, test_loader: DataLoader, **kwargs: Union[bool, str]):
    """
    Muestra los resultados de un modelo en un conjunto de imágenes

    Parameters
    ----------
    model : UNet
        Modelo a evaluar
    test_loader : DataLoader
        DataLoader con las imágenes a evaluar
    **kwargs : bool or str
        Argumentos adicionales para la generación de la gráfica:
        - save : (bool) Si se guardan las imágenes o se muestran
        - show_size : (int) Número de imágenes a mostrar
        - name : (str) Nombre del archivo a guardar
        - path : (str) Ruta donde se guardarán las imágenes
    """
    imgs = next(iter(test_loader))
    imgs = imgs.to(CUDA)
    model = model.to(CUDA)

    with torch.no_grad():
        model.eval()

        with autocast(device_type="cuda", dtype=torch.float16):
            scores = model(imgs)

        result = (scores > 0.5).float()

    imgs = imgs.cpu()
    result = result.cpu()
    plot_batch(
        imgs=imgs,
        masks=result,
        **kwargs
    )
