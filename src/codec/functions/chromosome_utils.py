"""
Módulo con funciones auxiliares para el manejo de cromosomas

Funciones
---------
- get_num_layers: Obtiene el número de capas activas de un cromosoma
- get_num_convs: Obtiene el número de convoluciones activas de una serie de convoluciones
"""
from typing import Union, Optional

from ..constants import (
    Encoding,

    LEN_CONV_REAL, LEN_CONVS_REAL, LEN_LAYER_REAL,
    LEN_POOLINGS_REAL, LEN_FILTERS_REAL,
    LEN_CONV_BIN, LEN_CONVS_BIN, LEN_LAYER_BIN,
    LEN_POOLINGS_BIN, LEN_FILTERS_BIN,

    IDENTITY_FILTERS_REAL, IDENTITY_FILTERS_BIN,
    IDENTITY_POOLING_REAL, IDENTITY_POOLING_BIN
)


def split_by_sizes(segment: list, sizes: list) -> list:
    """
    Divide un segmento en partes de tamaños específicos

    Parameters
    ----------
    segment : list
        Segmento a dividir
    sizes : list
        Lista de tamaños de cada parte

    Returns
    -------
    list
        Lista de partes divididas
    """
    result = []
    pointer = 0

    for size in sizes:
        result.append(segment[pointer:pointer + size])
        pointer += size

    return result


def layer_is_identity(layer: Union[Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
                                                        str],
                                                  tuple[list[Optional[tuple[int, int, str]]],
                                                        bool]]],
                                   list[float], str],
                      encoding: Union[Encoding, str]) -> bool:
    """
    Determina si una capa es una capa de identidad

    Parameters
    ----------
    layer : tuple or list or str
        Capa a evaluar
    encoding : Encoding or str
        Tipo de codificación del cromosoma (decodificado, real o binario)

    Returns
    -------
    bool
        `True` si la capa es una capa de identidad, `False` en caso contrario
    """
    encoding = Encoding(encoding)

    if encoding == Encoding.DECODED:
        if layer is None:
            return True

        encoder_convs, pooling = layer[0]
        decoder_convs, _ = layer[1]

        if ((get_num_convs(encoder_convs, encoding) == 0
            or get_num_convs(decoder_convs, encoding) == 0)
                and pooling == None):
            return True
    elif encoding == Encoding.REAL:
        encoder_convs, pooling, decoder_convs = split_by_sizes(
            layer,
            [LEN_CONVS_REAL, LEN_POOLINGS_REAL, LEN_CONVS_REAL]
        )

        if ((get_num_convs(encoder_convs, encoding) == 0
            or get_num_convs(decoder_convs, encoding) == 0)
                and pooling < IDENTITY_POOLING_REAL):
            return True
    elif encoding == Encoding.BINARY:
        encoder_convs, pooling, decoder_convs = split_by_sizes(
            layer,
            [LEN_CONVS_BIN, LEN_POOLINGS_BIN, LEN_CONVS_BIN]
        )

        if ((get_num_convs(encoder_convs, encoding) == 0
            or get_num_convs(decoder_convs, encoding) == 0)
                and pooling == IDENTITY_POOLING_BIN):
            return True

    return False


def conv_is_identity(conv: Union[Optional[tuple[int, int, str]], list[float], str],
                     encoding: Union[Encoding, str]) -> bool:
    """
    Determina si una convolución es una convolución de identidad

    Parameters
    ----------
    conv : tuple or list or str
        Convolución a evaluar
    encoding : Encoding or str
        Tipo de codificación del cromosoma (decodificado, real o binario)

    Returns
    -------
    bool
        `True` si la convolución es una convolución de identidad, `False` en caso contrario
    """
    encoding = Encoding(encoding)

    if encoding == Encoding.DECODED:
        if conv is None or conv[0] is None:
            return True
    elif encoding == Encoding.REAL:
        filters = conv[0:LEN_FILTERS_REAL]

        if all(x < y for x, y in zip(filters, IDENTITY_FILTERS_REAL)):
            return True
    elif encoding == Encoding.BINARY:
        filters = conv[0:LEN_FILTERS_BIN]

        if (filters == IDENTITY_FILTERS_BIN):
            return True

    return False


def get_num_layers(chromosome: Union[
        tuple[list[Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
                                        str],
                                  tuple[list[Optional[tuple[int, int, str]]],
                                        bool]]]],
              list[Optional[tuple[int, int, str]]]],
        list[float], str],
        encoding: Union[Encoding, str]) -> int:
    """
    Obtiene el número de capas activas de un cromosoma

    Parameters
    ----------
    chromosome : tuple or list or str
        Cromosoma a evaluar (decodificado, real o binario)
    encoding : Encoding or str
        Tipo de codificación del cromosoma (decodificado, real o binario)

    Returns
    -------
    int
        Número de capas activas
    """
    encoding = Encoding(encoding)

    num_layers = 0

    if encoding == Encoding.DECODED:
        for layer in chromosome[0]:
            if not layer_is_identity(layer, encoding):
                num_layers += 1
    elif encoding == Encoding.REAL:
        for i in range(0, len(chromosome) - LEN_CONVS_REAL, LEN_LAYER_REAL):
            if not layer_is_identity(chromosome[i:i + LEN_LAYER_REAL], encoding):
                num_layers += 1
    elif encoding == Encoding.BINARY:
        for i in range(0, len(chromosome) - LEN_CONVS_BIN, LEN_LAYER_BIN):
            if not layer_is_identity(chromosome[i:i + LEN_LAYER_BIN], encoding):
                num_layers += 1

    return num_layers


def get_num_convs(convs: Union[list[Optional[tuple[int, int, str]]],
                               list[float], str],
                  encoding: Union[Encoding, str]) -> int:
    """
    Obtiene el número de convoluciones activas de una seríe de convoluciones

    Parameters
    ----------
    convs : list or str
        Convoluciones a evaluar (decodificadas, reales o binarias)
    encoding : Encoding or str
        Tipo de codificación del cromosoma (decodificado, real o binario)

    Returns
    -------
    int
        Número de convoluciones activas
    """
    encoding = Encoding(encoding)

    num_convs = 0

    if encoding == Encoding.DECODED:
        for conv in convs:
            if not conv_is_identity(conv, encoding):
                num_convs += 1
    elif encoding == Encoding.REAL:
        for i in range(0, len(convs), LEN_CONV_REAL):
            if not conv_is_identity(convs[i:i + LEN_CONV_REAL], encoding):
                num_convs += 1
    elif encoding == Encoding.BINARY:
        for i in range(0, len(convs), LEN_CONV_BIN):
            if not conv_is_identity(convs[i:i + LEN_CONV_BIN], encoding):
                num_convs += 1

    return num_convs
