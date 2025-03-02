"""
Archivo temporal para pruebas de código
"""

# ESTRUCTURA DE UN CROMOSOMA
tipo: tuple[list[tuple[tuple[list[tuple[int, int, str]],
                             str],
                       tuple[list[tuple[int, int, str]],
                             bool]]],
            list[tuple[int, int, str]]]
decoded = (
    # chromosome: [layers, bottleneck]
    [  # layers: [convs+deconvs, convs+deconvs, ...]
        (  # convs+deconvs: [nconvs+pooling, nconvs+concat]
            (  # nconvs+pooling: [nconvs, pooling]
                [  # nconvs: [conv, conv, ...]
                    (3, 2, "a"),  # conv: [f, s, a]
                    (3, 2, "a")
                ],
                # pooling
                "p"
            ),
            (  # nconvs+concat: [nconvs, concat]
                [  # nconvs: [conv, conv, ...]
                    (3, 2, "a"),  # conv: [f, s, a]
                    (3, 2, "a")
                ],
                # concat
                True
            )
        ),
        (
            (
                [
                    (3, 2, "a"),
                    (3, 2, "a")
                ],
                "p"
            ),
            (
                [
                    (3, 2, "a"),
                    (3, 2, "a")
                ],
                False
            )
        ),
    ],
    [  # bottleneck: [conv, conv, ...]
        (3, 2, "a"),  # conv: [f, s, a]
        (3, 2, "a")
    ]
)


# %% Prueba de DataLoader
from torch_utils import CARVANA_BATCH_SIZE, ROAD_BATCH_SIZE, CAR_BATCH_SIZE
from torch_utils import TorchDataLoader
from torch_utils import plot_batch

# Elige el dataset a cargar
# name = "carvana"
# data_path = "../carvana-dataset/"
# bs = CARVANA_BATCH_SIZE
# name = "road"
# bs = ROAD_BATCH_SIZE
# data_path = "../road-dataset/"
name = "car"
bs = CAR_BATCH_SIZE
data_path = "../car-dataset/"

data_loader = TorchDataLoader(name, data_path=data_path)
imgs, masks = next(iter(data_loader.train))

# SHOW_SIZE solo es para mostrar las imágenes de prueba, si se quiere ver un batch de entrenamiento
# se debe asignar show_size al correspondiente del dataset, por ejemplo, CARVANA_BATCH_SIZE o ROAD_BATCH_SIZE
plot_batch(imgs, masks, show_size=bs)

# Comprobación de las dimensiones de los DataLoaders
train = data_loader.train
data_train = next(iter(train))
val = data_loader.validation
data_val = next(iter(val))
test = data_loader.test
data_test = next(iter(test))

print(
    f"Train: {len(train)},\t length: {len(data_train)},\t shape: {data_train[0].shape}"
)
print(
    f"Val:   {len(val)},\t length: {len(data_val)},\t shape: {data_val[0].shape}"
)
print(
    f"Test:  {len(test)},\t length: 1,\t shape: {data_test.shape}"
)


# %% Prueba de codificación y decodificación
from codec import Chromosome

unet_paper = (
    [  # layers: [convs+deconvs, convs+deconvs, ...]
        (  # convs+deconvs: [nconvs+pooling, nconvs+concat]
            (  # nconvs+pooling: [nconvs, pooling]
                [  # nconvs: [conv, conv, ...]
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                "max"  # pooling
            ),
            (  # nconvs+concat: [nconvs, concat]
                [  # nconvs: [conv, conv, ...]
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                True  # concat
            )
        ),
        (
            (
                [
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                True
            )
        )
    ],
    [  # bottleneck: [conv, conv, ...]
        (1024, 3, "relu"),  # conv: [f, s, a]
        (1024, 3, "relu")
    ]
)

c = Chromosome(
    max_layers=4,
    max_conv_per_layer=2,
    chromosome=unet_paper
)

print("Name:\n", c)
print("Decoded:\n", c.get_decoded())
print("Real:\n", c.get_real())
print("Binary:\n", c.get_binary())

# Comprobamos que la decodificación y codificación sean correctas
assert c.get_decoded() == unet_paper
# Aunque tenga letras se trata de una codificación binaria
# Solo está comprimida para que no sea tan larga
assert (
    c.get_binary(zip=True) == "ARARAIQIQ4IMIGEGEHKDKBVBVB6Q6QPIPIPEHEA_188"
)


# %% Prueba de entrenamiento y evaluación
from codec import Chromosome

# Distintas redes a elegir
unet_paper = "ARARAIQIQ4IMIGEGEHKDKBVBVB6Q6QPIPIPEHEA_188"
unet_paper_mini = "AJAJAEQEQWIGIDEDEF2B2A5A5BKQKQFIFIKECEA_188"
unet_rara = "AE6HZLHCTEYFIMM24G3TPQWZ4AS5CUI_146"
c = Chromosome(
    max_layers=4,
    max_conv_per_layer=2,
    chromosome=unet_paper
)

# Mostramos la arquitectura a generar
print(c.get_decoded())

# Al entrenarla se generará el modelo UNet automáticamente
c.train_unet(
    data_loader="car",
    epochs=1,
    data_path="../car-dataset/"
)

# Si no especificamos un DataLoader, se usará el dataloader con el que se entrenó
c.show_results()
# Aunque se haya entrenado con un DataLoader, podemos evaluar con otro
c.show_results("carvana", data_path="../carvana-dataset/")

# Tampoco es necesario especificar el DataLoader si ya se ha entrenado
print("Aptitud para el DataLoader de entrenamiento:", c.get_aptitude())
print("Aptitud para el DataLoader de validación:", c.get_aptitude(
    "carvana", data_path="../carvana-dataset/"
))

# Guardamos los resultados
c.show_results(
    save=True
)
c.show_results(
    "carvana",
    save=True,
    name="para carvana.png",
    data_path="../carvana-dataset/"
)

c.save_unet()

# %% Prueba de congruencia entre discretización y decodificación
FILTERS = {
    '0000': 0,
    '0001': 2**0,
    '0011': 2**1,
    '0010': 2**2,
    '0110': 2**3,
    '0111': 2**4,
    '0101': 2**5,
    '0100': 2**6,
    '1100': 2**7,
    '1101': 2**8,
    '1111': 2**9,
    '1110': 2**10,
}
KERNEL_SIZES = {
    '00': 1,
    '01': 3,
    '11': 5,
}
ACTIVATION_FUNCTIONS = {
    '0000': 'relu',
    '0001': 'sigmoid',
    '0011': 'tanh',
    '0010': 'softmax',
    '0110': 'softplus',
    '0111': 'softsign',
    '0101': 'selu',
    '0100': 'elu',
    '1100': 'exponential',
    '1101': 'linear',
}
POOLINGS = {
    '0': 'max',
    '1': 'average',
}
CONCATENATION = {
    '0': False,
    '1': True,
}


def discretize_gene(value: int | bool | str, options: dict) -> str:
    """
    Devuelve el valor real equivalente a value en las opciones

    Parameters
    ----------
    value : int or bool or str
        Valor a discretizar
    options : dict
        Opciones de discretización

    Returns
    -------
    El valor encontrado en esa lista

    Raises
    -------
    ValueError
        Si el valor no está en la lista
    """
    step = 1 / len(options)
    real_rep = 0.01

    for _, v in options.items():
        if v == value:
            return round(real_rep, 2)

        real_rep += step

    raise ValueError(f"Value {value} not found in options")


def decode_gene(value: float, options: dict) -> int | bool | str:
    """
    Para real:
        Pasa el tamaño del diccionario a un rango de 0 a 1, selecciona el valor equivalente a value y lo devuelve
    Para binario:
        Devuelve el valor equivalente a value en el diccionario

    Parameters
    ----------
    value : float
        Valor a decodificar
    options : dict
        Opciones de decodificación

    Returns
    -------
    El valor encontrado en esa lista
    """
    step = 1 / len(options)

    for i in range(len(options)):
        cota_inf = step * i
        cota_sup = step * (i + 1)

        if cota_inf <= value < cota_sup:
            return list(options.values())[i]

    return list(options.values())[-1]


def probarDic(dic, dic_str):
    keys = list(dic.keys())
    for key in keys:
        print(f"Probando {dic_str}[{key}]: {dic[key]} -> ", end="")
        real = discretize_gene(dic[key], dic)
        decode = decode_gene(real, dic)
        print(f"{dic_str}[{real}] = {decode}")


probarDic(FILTERS, "FILTERS")
probarDic(KERNEL_SIZES, "KERNEL_SIZES")
probarDic(ACTIVATION_FUNCTIONS, "ACTIVATION_FUNCTIONS")
probarDic(POOLINGS, "POOLINGS")
probarDic(CONCATENATION, "CONCATENATION")


# %%
encoded_conv = ['0000', '01', '0010']
encoded_conv = "".join(encoded_conv)
print(encoded_conv)


# %%
def validar_lista_booleana(valores) -> bool:
    return all(v in [0, 1] for v in valores)


print(validar_lista_booleana([2, False]))  # True
print(validar_lista_booleana([0, 1]))      # False

# %%

a = ['1000', '01', '0010']
b = "123"

value = "".join(map(str, b[0:2]))
print(value)
value = "".join(b[0:2])
print(value)
value = b[0:2]
print(value)

encoded_conv = "".join(a)
print(encoded_conv)

# %%

encoded_chromosome = "101"
result = "010"
encoded_chromosome += result  # Concatenación de strings
print(encoded_chromosome)  # Salida: "101010"

# %%
binary = "01011"
if not all(v in ['0', '1'] for v in binary):
    print("error")

# %%
binary = "01011"
not_binary = ""

if not_binary:
    print("si")

# %%
import random
random.seed(1)

random.randint(2, 200)

# %%


def probar_str(var):
    print(type(var))
    var += "Hola"
    print(var)
    print(type(var))


_a = ""
_b = str()

probar_str(_b)

# %%

a = tuple()

print(type(a))


# %%
import random

a = random.randint(2, 2)
print(a)


# %%
from PIL import Image
import sys
import numpy as np


def obtener_shape_imagen(ruta_imagen):
    try:
        imagen = Image.open(ruta_imagen).convert("L")
        imagen_array = np.array(imagen)
        print(
            f"Shape de la imagen: {imagen_array.shape} (alto, ancho, canales)")

        import matplotlib.pyplot as plt
        plt.imshow(imagen_array, cmap="gray")
        plt.show()
    except Exception as e:
        print(
            f"Error: No se pudo cargar la imagen. Verifica la ruta. Detalles: {e}")
        sys.exit(1)


if __name__ == "__main__":
    ruta = "./road-dataset/masks/02_00_000.png.png"
    obtener_shape_imagen(ruta)


# %%
class A:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def __str__(self):
        return f"{self.a}, {self.b}"


clases = {
    "A": A()
}

instancia = clases["A"]
print(instancia)
print()
instancia2 = clases["A"]
print(instancia)
print(instancia2)
print()
instancia2.a = 5
instancia2.b = 6
print(instancia)
print(instancia2)


# %%
def func(a, b):
    print(a + b)


uno = True
func(
    1,
    1
    if uno else
    2
)


# %%
import torch

x = torch.randn(4, 4)
print(x)
print(x.shape)

y = x.view(-1)
print(y)
print(y.shape)


# %%
def prueba(a, **_):
    print(a)


def prueba2(**kwargs):
    prueba(**kwargs)


prueba(a=1, b=2)


# %%
import base64


def encode_binary(binary_string: str) -> str:
    # Guarda la longitud original
    length = len(binary_string)
    # Convierte la cadena binaria en bytes
    byte_data = int(binary_string, 2).to_bytes(
        (length + 7) // 8, byteorder='big')
    # Codifica en base32 y adjunta la longitud
    return f"{base64.b32encode(byte_data).decode().rstrip('=')}:{length}"


def decode_binary(encoded_string: str) -> str:
    # Separa la parte codificada y la longitud original
    encoded_data, length = encoded_string.split(":")
    length = int(length)
    # Añade padding necesario para Base32 y decodifica
    padding = '=' * ((8 - len(encoded_data) % 8) % 8)
    byte_data = base64.b32decode(encoded_data + padding)
    # Convierte los bytes de vuelta a binario y recorta los ceros extra
    return bin(int.from_bytes(byte_data, byteorder='big'))[2:].zfill(length)


# Ejemplo de uso
binary_string = "01001111000111110010101100111000101001100100110000010101000011000110011010111000011011011100110111110000101101100111100000001001011101000101010001"
coded = encode_binary(binary_string)
decoded = decode_binary(coded)

print("Original:", binary_string)
print("Codificado:", coded)
print("Decodificado:", decoded)

assert binary_string == decoded


# %%
import torch

a = torch.tensor([1])

print(1 - a)


# %%
kwargs = {
    "data_path": "path1",
    "other_key": "value"
}

data_loader_args = {
    "transform": "transform2",
    "data_path": "path2"
}

data_loader_args = {
    k: v for k, v in kwargs.items()
    if k in ["transform", "data_path"] and (k not in data_loader_args or v is not None)
}


print(data_loader_args)

# %%
