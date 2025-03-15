import base64
from typing import Union, Optional

from .constants import (
    REAL_POOLING_LEN, REAL_CONCAT_LEN,
    REAL_CONVS_LEN, REAL_LAYER_LEN,
    BIN_CONVS_LEN, BIN_LAYER_LEN,

    BIN_POOLING_LEN, BIN_CONCAT_LEN,
    IDENTITY_CONV_REAL, IDENTITY_LAYER_REAL,
    IDENTITY_CONV_BIN, IDENTITY_LAYER_BIN,

    FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS,
    POOLINGS, CONCATENATION
)


def unzip_binary(binary: str) -> str:
    """
    Decodifica un string binario en base32

    Parameters
    ----------
    binary : str
        String binario a decodificar

    Returns
    -------
    str
        String decodificado

    Raises
    ------
    ValueError
        Si el cromosoma binario no está en el formato correcto
    """
    if '_' not in binary:
        return binary

    binary = binary.split("_")

    if len(binary) != 2:
        raise ValueError(
            "El cromosoma binario no está en el formato correcto"
        )

    encoded_data, length = binary
    length = int(length)
    padding = '=' * ((8 - len(encoded_data) % 8) % 8)
    byte_data = base64.b32decode(encoded_data + padding)

    return bin(int.from_bytes(byte_data, byteorder='big'))[2:].zfill(length)


def decode_gene(value: Union[list[float], str], options: dict[str, Union[int, bool, str]],
                real: bool) -> Union[int, bool, str]:
    """
    Para real:
        Pasa el tamaño del diccionario a un rango de 0 a 1, selecciona el valor
        equivalente a value y lo devuelve
    Para binario:
        Devuelve el valor en la posición 'value' del diccionario

    Parameters
    ----------
    value : float or str
        Valor a decodificar
    options : dict
        Opciones de decodificación
    real : bool
        Indica si la codificación del gen es real o binaria

    Returns
    -------
    int or bool or str
        Valor encontrado en las opciones
    """
    if not real:
        if value in options:
            return options[value]
        else:
            raise ValueError(
                f"El valor {value} no está en las opciones"
            )
    else:
        value = value[0] if isinstance(value, list) else value
        step = 1 / len(options)

        for i in range(len(options)):
            cota_inf = step * i
            cota_sup = step * (i + 1)

            if cota_inf <= value < cota_sup:
                return list(options.values())[i]

        return list(options.values())[-1]


def decode_convs(convs: Union[list[float], str],
                 real: bool) -> list[Optional[tuple[int, int, str]]]:
    """
    Decodifica una lista de convoluciones

    Parameters
    ----------
    convs : list or str
        Lista de convoluciones
    real : bool
        Indica si la codificación de las convoluciones es real o binaria

    Returns
    -------
    list
        Lista de convoluciones decodificadas
    """
    decoded_convs = []

    for i in range(0, len(convs), 3 if real else 10):
        if ((real and convs[i:i + 3] == IDENTITY_CONV_REAL)
                or (not real and convs[i:i + 10] == IDENTITY_CONV_BIN)):
            decoded_convs.append(None)
        else:
            decoded_convs.append(
                (
                    decode_gene(  # f
                        value=convs[i] if real else convs[i:i + 4],
                        options=FILTERS,
                        real=real
                    ),
                    decode_gene(  # s
                        value=convs[i + 1] if real else convs[i + 4:i + 6],
                        options=KERNEL_SIZES,
                        real=real
                    ),
                    decode_gene(  # a
                        value=convs[i + 2] if real else convs[i + 6:i + 10],
                        options=ACTIVATION_FUNCTIONS,
                        real=real
                    )
                )
            )

    return decoded_convs


def decode_layer(layer: Union[list[float], str],
                 real: bool) -> Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
                                                     str],
                                               tuple[list[Optional[tuple[int, int, str]]],
                                                     bool]]]:
    """
    Decodifica una capa del cromosoma

    Parameters
    ----------
    layer : list or str
        Capa a decodificar
    real : bool
        Indica si la codificación de la capa es real o binaria

    Returns
    -------
    tuple
        Capa decodificada
    """
    if ((real and layer == IDENTITY_LAYER_REAL)
            or (not real and layer == IDENTITY_LAYER_BIN)):
        return None

    if real:
        pooling_len = REAL_POOLING_LEN
        concat_len = REAL_CONCAT_LEN
        encoder_len = REAL_CONVS_LEN + REAL_POOLING_LEN
    else:
        pooling_len = BIN_POOLING_LEN
        concat_len = BIN_CONCAT_LEN
        encoder_len = BIN_CONVS_LEN + BIN_POOLING_LEN

    encoder = layer[:encoder_len]
    decoder = layer[encoder_len:]

    pooling = decode_gene(
        value=encoder[-pooling_len:],
        options=POOLINGS,
        real=real
    )
    concat = decode_gene(
        value=decoder[-concat_len:],
        options=CONCATENATION,
        real=real
    )

    # Decodificamos encoder
    decoded_convolutions = decode_convs(
        convs=encoder[0:len(encoder) - pooling_len],
        real=real
    )
    # Decodificamos decoder
    decoded_deconvolutions = decode_convs(
        convs=decoder[0:len(decoder) - concat_len],
        real=real
    )

    return ((decoded_convolutions, pooling), (decoded_deconvolutions, concat))


def decode_chromosome(
    chromosome: Union[list[float], str],
    real: bool
) -> tuple[list[Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
                                     str],
                               tuple[list[Optional[tuple[int, int, str]]],
                                     bool]]]],
           list[Optional[tuple[int, int, str]]]]:
    """
    Transforma el cromosoma en una lista con los valores de las capas, entendible para el humano y
    siendo un paso previo a la creación del modelo

    Parameters
    ----------
    chromosome : list or str
        Cromosoma a decodificar
    layer_len : int
        Longitud de una capa
    bottleneck_len : int
        Longitud del cuello de botella
    real : bool
        Indica si la codificación del cromosoma es real o binaria

    Returns
    -------
    tuple
        Cromosoma decodificado
    """
    if real:
        layer_len = REAL_LAYER_LEN
        bottleneck_len = REAL_CONVS_LEN
    else:
        layer_len = BIN_LAYER_LEN
        bottleneck_len = BIN_CONVS_LEN

    decoded_layers = [
        decode_layer(
            layer=chromosome[i:i + layer_len],
            real=real
        )
        for i in range(0, len(chromosome) - bottleneck_len, layer_len)
    ]

    decoded_bottleneck = decode_convs(
        convs=chromosome[len(chromosome) - bottleneck_len:],
        real=real
    )

    return (decoded_layers, decoded_bottleneck)
