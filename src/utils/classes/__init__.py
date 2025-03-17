"""
Módulo con clases útiles para la implementación de modelos de segmentación de imágenes

Clases
------
- UNet: Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado
- TorchDataLoader: Wrapper de los DataLoaders de train, validation y test para un TorchDataset
"""
from .unet import UNet
from .dataloader import TorchDataLoader

__all__ = [
    "UNet",
    "TorchDataLoader"
]
