"""
Módulo con funciones útiles para el manejo de los cromosomas

Funciones
---------
- zip_binary: Codifica un string binario en base32
- encode_chromosome: Codifica un cromosoma a binario o real
- unzip_binary: Decodifica un string binario en base32
- decode_chromosome: Decodifica un cromosoma a binario o real
- get_num_layers: Obtiene el número de capas activas de un cromosoma
- get_num_convs: Obtiene el número de convoluciones activas de una serie de convoluciones
- layer_is_identity: Determina si una capa es una capa de identidad
- conv_is_identity: Determina si una convolución es una convolución de identidad
"""
from .encode import zip_binary, encode_chromosome
from .decode import unzip_binary, decode_chromosome
from .chromosome_utils import (
    get_num_layers, get_num_convs,
    layer_is_identity, conv_is_identity
)

__all__ = [
    "zip_binary", "encode_chromosome",

    "unzip_binary", "decode_chromosome",

    "get_num_layers", "get_num_convs",
    "layer_is_identity", "conv_is_identity"
]
