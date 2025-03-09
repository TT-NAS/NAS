import base64
from typing import Union

from .constants import FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS, POOLINGS, CONCATENATION


def zip_binary(binary: str) -> str:
    """
    Codifica un string binario en base32 y lo devuelve junto con su longitud

    Parameters
    ----------
    binary : str
        String binario a codificar

    Returns
    -------
    str
        String codificado en base32 y su longitud
    """
    length = len(binary)
    byte_data = int(binary, 2).to_bytes(
        (length + 7) // 8,
        byteorder='big'
    )

    return f"{base64.b32encode(byte_data).decode().rstrip('=')}_{length}"


def encode_gene(value: Union[int, bool, str], options: dict, real: bool) -> Union[float, str]:
    """
    Devuelve la codificación equivalente a value en las opciones

    Parameters
    ----------
    value : int or bool or str
        Valor a codificar
    options : dict
        Diccionario de valores y su codificación
    real : bool
        Indica si la codificación deseada es real o binaria

    Returns
    -------
    float or str
        Valor codificado en
        - un flotante con rango [0, 1] si `real=True`
        - un string con 0s y 1s si `real=False`

    Raises
    -------
    ValueError
        Si el valor no está en las opciones
    """
    if real:
        step = 1 / len(options)
        real_representation = 0.01

        for _, v in options.items():
            if v == value:
                return round(real_representation, 2)

            real_representation += step

    else:
        for k, v in options.items():
            if v == value:
                return k

    raise ValueError(f"El valor {value} no está en las opciones")


def encode_convs(convs: list[tuple[int, int, str]], real: bool) -> Union[list[float], str]:
    """
    Codifica una lista de convoluciones

    Parameters
    ----------
    convs : list
        Lista con los valores de las convoluciones
    real : bool
        Indica si la codificación deseada es real o binaria

    Returns
    -------
    list or str
        Convoluciones codificadas en
        - una lista de flotantes si `real=True`
        - un string si `real=False`
    """
    encoded_convs = [] if real else ""

    for f, s, a in convs:
        encoded_conv = [
            encode_gene(
                value=f,
                options=FILTERS,
                real=real
            ),
            encode_gene(
                value=s,
                options=KERNEL_SIZES,
                real=real
            ),
            encode_gene(
                value=a,
                options=ACTIVATION_FUNCTIONS,
                real=real
            )
        ]

        if real:
            encoded_convs.extend(encoded_conv)
        else:
            encoded_conv = "".join(encoded_conv)
            encoded_convs += encoded_conv

    return encoded_convs


def encode_chromosome(
    decoded_chromosome:
        tuple[list[tuple[tuple[list[tuple[int, int, str]],
                               str],
                         tuple[list[tuple[int, int, str]],
                               bool]]],
              list[tuple[int, int, str]]],
    real: bool
) -> Union[list[float], str]:
    """
    Devuelve un cromosoma en codificación real o binaria

    Parameters
    ----------
    decoded_chromosome : tuple
        Cromosoma decodificado
    real : bool
        Indica si la codificación deseada es real o binaria

    Returns
    -------
    list or str
        Cromosoma codificado en
        - una lista de flotantes si `real=True`
        - un string si `real=False`
    """
    encoded_chromosome = [] if real else str()

    for encoder, decoder in decoded_chromosome[0]:
        encoded_encoder = encode_convs(encoder[0], real=real)
        encoded_pooling = encode_gene(
            value=encoder[1],
            options=POOLINGS,
            real=real
        )
        encoded_decoder = encode_convs(decoder[0], real=real)
        encoded_concat = encode_gene(
            value=decoder[1],
            options=CONCATENATION,
            real=real
        )

        if real:
            encoded_chromosome.extend(encoded_encoder)
            encoded_chromosome.append(encoded_pooling)
            encoded_chromosome.extend(encoded_decoder)
            encoded_chromosome.append(encoded_concat)

        else:
            encoded_chromosome += encoded_encoder
            encoded_chromosome += encoded_pooling
            encoded_chromosome += encoded_decoder
            encoded_chromosome += encoded_concat

    encoded_botleneck = encode_convs(decoded_chromosome[1], real=real)

    if real:
        encoded_chromosome.extend(encoded_botleneck)
    else:
        encoded_chromosome += encoded_botleneck

    return encoded_chromosome
