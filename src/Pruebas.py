"""
Archivo temporal para pruebas de código
"""
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
    max_convs_per_layer=2,
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
    c.get_binary(zip=True) == "AVCVCKRKRUIUISEKEPCHCFRDRD2R2RNI5I7UPUI_188"
)


# %% Prueba de entrenamiento y evaluación
from codec import Chromosome

# Distintas redes a elegir
unet_paper = "AVCVCKRKRUIUISEKEPCHCFRDRD2R2RNI5I7UPUI_188"
c = Chromosome(
    max_layers=4,
    max_convs_per_layer=2,
    chromosome=unet_paper
)

# Mostramos la arquitectura a generar
print(c.get_decoded())

data_loader = "carvana"
data_path = "../carvana-dataset/"
data_loader_alternativo = "car"
data_path_alternativo = "../car-dataset/"

# Al entrenarla se generará el modelo UNet automáticamente
c.train_unet(
    data_loader=data_loader,
    epochs=1,
    data_path=data_path,
    dataset_len=2000
)

# Si no especificamos un DataLoader, se usará el dataloader con el que se entrenó
c.show_results()
# Aunque se haya entrenado con un DataLoader, podemos evaluar con otro
c.show_results(data_loader_alternativo, data_path=data_path_alternativo)

# Tampoco es necesario especificar el DataLoader si ya se ha entrenado
print("Aptitud para el DataLoader de entrenamiento:", c.get_aptitude())
print("Aptitud para el otro DataLoader:", c.get_aptitude(
    data_loader=data_loader_alternativo,
    data_path=data_path_alternativo
))

# Guardamos los resultados
c.show_results(
    save=True
)
c.show_results(
    data_loader_alternativo,
    save=True,
    name="para el otro DataLoader.png",
    data_path=data_path_alternativo
)

c.save_unet()

# %% Prueba de congruencia entre discretización y decodificación
FILTERS = {
    '0000': 2**0,  # 1
    '0001': 2**1,  # 2
    '0011': 2**2,  # 4
    '0010': 2**3,  # 8
    '0110': 2**4,  # 16
    '0111': 2**5,  # 32
    '0101': 2**6,  # 64
    '0100': 2**7,  # 128
    '1100': 2**8,  # 256
    '1101': 2**9,  # 512
    '1111': 2**10,  # 1024
}
KERNEL_SIZES = {
    '00': 1,
    '01': 3,
    '11': 5,
}
ACTIVATION_FUNCTIONS = {
    '0000': 'linear',
    '0001': 'relu',
    '0011': 'softplus',
    '0010': 'elu',
    '0110': 'selu',
    '0111': 'sigmoid',
    '0101': 'tanh',
    '0100': 'softsign',
    '1100': 'softmax'
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
