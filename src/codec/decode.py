import base64
from typing import Union

from .constants import FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS, POOLINGS, CONCATENATION


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


def decode_gene(value: Union[float, str], options: dict[str, Union[int, bool, str]],
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
            return options[list(options.keys())[-1]]
    else:
        step = 1 / len(options)

        for i in range(len(options)):
            cota_inf = step * i
            cota_sup = step * (i + 1)

            if cota_inf <= value < cota_sup:
                return list(options.values())[i]

        return list(options.values())[-1]


def decode_convs(convs: Union[list[float], str], real: bool) -> list[tuple[int, int, str]]:
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
    return [
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
        for i in range(0, len(convs), 3 if real else 10)
    ]


def decode_layer(layer: Union[list[float], str],
                 real: bool) -> tuple[tuple[list[tuple[int, int, str]],
                                            str],
                                      tuple[list[tuple[int, int, str]],
                                            bool]]:
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
    encoder = layer[:len(layer) // 2]
    decoder = layer[len(layer) // 2:]

    pooling = decode_gene(
        value=encoder[-1],
        options=POOLINGS,
        real=real
    )
    concat = decode_gene(
        value=decoder[-1],
        options=CONCATENATION,
        real=real
    )

    # Decodificamos encoder
    decoded_convolutions = decode_convs(
        convs=encoder[0:len(encoder) - 1],
        real=real
    )
    # Decodificamos decoder
    decoded_deconvolutions = decode_convs(
        convs=decoder[0:len(decoder) - 1],
        real=real
    )

    return ((decoded_convolutions, pooling), (decoded_deconvolutions, concat))


def decode_chromosome(
    chromosome: Union[list[float], str],
    layer_len: int,
    bottleneck_len: int,
    real: bool
) -> tuple[list[tuple[tuple[list[tuple[int, int, str]],
                            str],
                      tuple[list[tuple[int, int, str]],
                            bool]]],
           list[tuple[int, int, str]]]:
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
    decoded_layers = [
        decode_layer(
            layer=chromosome[i:i + layer_len],
            real=real
        )
        for i in range(0, len(chromosome) - bottleneck_len, layer_len)
    ]

    decoded_bottleneck = decode_convs(
        convs=chromosome[len(chromosome) - bottleneck_len:len(chromosome)],
        real=real
    )

    return (decoded_layers, decoded_bottleneck)
