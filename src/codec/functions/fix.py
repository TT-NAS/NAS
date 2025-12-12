"""
Módulo con funciones para corregir un cromosoma binario

Funciones
---------
- fix_gene: Corrige un gen
- fix_convs: Corrige una lista de convoluciones
- fix_layer: Corrige una capa
- fix_chromosome: Corrige un cromosoma a binario
"""
import random

from ..constants import (
    LEN_POOLINGS_BIN, LEN_CONCAT_BIN,
    LEN_CONVS_BIN, LEN_LAYER_BIN,

    IDENTITY_CONV_BIN,

    FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS,
    POOLINGS, CONCATENATION,

    INVALID_FILTERS_EQS, INVALID_KERNEL_SIZES_EQS,
    INVALID_ACTIVATION_FUNCTIONS_EQS, INVALID_POOLINGS_EQS
)


def fix_bin_gene(value: str, options: dict, equivalences: dict) -> str:
    """
    Devuelve el valor en la posición 'value' del diccionario,
    o su equivalencia si el valor está dañado.

    La elección de la equivalencia se realiza de forma aleatoria,
    mientras más cercana esté la llave de la equivalencia según
    la distancia Hamming, más probable es que sea elegida.

    Parameters
    ----------
    value : str
        Valor a corregir
    options : dict
        Opciones de decodificación
    equivalences : dict
        Tabla de equivalencias para las opciones de decodificación

    Returns
    -------
    str
        Valor corregido del gen
    """
    if value in options:
        return value
    else:
        return random.choices(
            population=equivalences["valid_filters"],
            weights=equivalences["weights"][value],
            k=1
        )[0]


def fix_bin_convs(convs: str) -> str:
    """
    Corrige una lista de convoluciones

    Parameters
    ----------
    convs : str
        Lista de convoluciones

    Returns
    -------
    str
        Convoluciones corregidas
    """
    fixed_convs = ""

    for i in range(0, len(convs), len(IDENTITY_CONV_BIN)):
        fixed_convs += fix_bin_gene(  # filters (f)
            value=convs[i:i + 4],
            options=FILTERS,
            equivalences=INVALID_FILTERS_EQS
        )
        fixed_convs += fix_bin_gene(  # kernel_size (s)
            value=convs[i + 4:i + 6],
            options=KERNEL_SIZES,
            equivalences=INVALID_KERNEL_SIZES_EQS
        )
        fixed_convs += fix_bin_gene(  # activation func (a)
            value=convs[i + 6:i + 10],
            options=ACTIVATION_FUNCTIONS,
            equivalences=INVALID_ACTIVATION_FUNCTIONS_EQS
        )

    return fixed_convs


def fix_bin_chromosome(chromosome: str) -> str:
    """
    Corrige un cromosoma binario completo

    Parameters
    ----------
    chromosome : str
        Cromosoma a corregir

    Returns
    -------
    str
        Cromosoma binario corregido
    """
    fixed_layers = ""

    for i in range(0, len(chromosome) - LEN_CONVS_BIN, LEN_LAYER_BIN):
        layer = chromosome[i:i + LEN_LAYER_BIN]
        encoder = layer[:LEN_CONVS_BIN + LEN_POOLINGS_BIN]
        decoder = layer[LEN_CONVS_BIN + LEN_POOLINGS_BIN:]

        pooling = fix_bin_gene(
            value=encoder[-LEN_POOLINGS_BIN:],
            options=POOLINGS,
            equivalences=INVALID_POOLINGS_EQS
        )
        concat = fix_bin_gene(
            value=decoder[-LEN_CONCAT_BIN:],
            options=CONCATENATION,
            equivalences={}
        )

        # Corregimos encoder
        fixed_convolutions = fix_bin_convs(
            convs=encoder[0:len(encoder) - LEN_POOLINGS_BIN]
        )
        # Corregimos decoder
        fixed_deconvolutions = fix_bin_convs(
            convs=decoder[0:len(decoder) - LEN_CONCAT_BIN]
        )

        fixed_layers += fixed_convolutions + pooling + fixed_deconvolutions + concat

    fixed_bottleneck = fix_bin_convs(
        convs=chromosome[len(chromosome) - LEN_CONVS_BIN:]
    )

    return fixed_layers + fixed_bottleneck
