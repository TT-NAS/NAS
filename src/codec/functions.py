from typing import Union, Optional

from .constants import (
    REAL_CONV_LEN, REAL_CONVS_LEN, REAL_LAYER_LEN,
    BIN_CONV_LEN, BIN_CONVS_LEN, BIN_LAYER_LEN,

    IDENTITY_CONV_REAL, IDENTITY_LAYER_REAL,
    IDENTITY_CONV_BIN, IDENTITY_LAYER_BIN
)


def get_num_layers(chromosome: Union[tuple, list, str]) -> int:
    num_layers = 0

    if isinstance(chromosome, tuple):  # decoded
        for layer in chromosome[0]:
            if layer is not None:
                num_layers += 1
    elif isinstance(chromosome, list):  # real
        for i in range(0, len(chromosome) - REAL_CONVS_LEN, REAL_LAYER_LEN):
            if chromosome[i:i + REAL_LAYER_LEN] != IDENTITY_LAYER_REAL:
                num_layers += 1
    elif isinstance(chromosome, str):  # binary
        for i in range(0, len(chromosome) - BIN_CONVS_LEN, BIN_LAYER_LEN):
            if chromosome[i:i + BIN_LAYER_LEN] != IDENTITY_LAYER_BIN:
                num_layers += 1

    return num_layers


def get_num_convs(convs: Union[list, str], decoded: Optional[bool] = None) -> int:
    num_convs = 0

    if isinstance(convs, str):  # binary
        for i in range(0, len(convs), BIN_CONV_LEN):
            if convs[i:i + BIN_CONV_LEN] != IDENTITY_CONV_BIN:
                num_convs += 1
    elif not decoded:  # real
        for i in range(0, len(convs), REAL_CONV_LEN):
            if convs[i:i + REAL_CONV_LEN] != IDENTITY_CONV_REAL:
                num_convs += 1
    else:  # decoded
        for conv in convs:
            if conv is not None:
                num_convs += 1

    return num_convs
