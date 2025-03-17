"""
Módulo con funciones para el manejo del estado del sistema

Funciones
---------
- empty_cache_torch: Limpia la memoria de la GPU de PyTorch
- set_current_net_binary: Establece la codificación binaria del modelo entrenandose actualmente
"""
import torch

import utils.globals as g


def empty_cache_torch():
    """
    Limpia la memoria de la GPU de PyTorch
    """
    torch.cuda.empty_cache()


def set_current_net_binary(net_binary: str):
    """
    Establece el nombre del modelo actual

    Parameters
    ----------
    net_binary : str
        Arquitectura del modelo en codificación binaria (`zip=True`)
    """
    global g

    g.CURRENT_NET_BINARY = net_binary
    a=3
