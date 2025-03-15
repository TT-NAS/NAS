"""
Archivo temporal para pruebas de código
"""
# %% Prueba de DataLoader
from utils import COCO_BATCH_SIZE, CARVANA_BATCH_SIZE, ROAD_BATCH_SIZE, CAR_BATCH_SIZE
from utils import COCO_PEOPLE_DATA_PATH, COCO_CAR_DATA_PATH, CARVANA_DATA_PATH, ROAD_DATA_PATH, CAR_DATA_PATH
from utils import TorchDataLoader
from utils import plot_batch

# Elige el dataset a cargar
name = "coco-people"
bs = COCO_BATCH_SIZE
data_path = COCO_PEOPLE_DATA_PATH
# name = "coco-car"
# bs = COCO_BATCH_SIZE
# data_path = COCO_CAR_DATA_PATH
# name = "carvana"
# bs = CARVANA_BATCH_SIZE
# data_path = CARVANA_DATA_PATH
# name = "road"
# bs = ROAD_BATCH_SIZE
# data_path = ROAD_DATA_PATH
# name = "car"
# bs = CAR_BATCH_SIZE
# data_path = CAR_DATA_PATH

data_path = "." + data_path

data_loader = TorchDataLoader(name, data_path=data_path)
imgs, masks = next(iter(data_loader.train))

# SHOW_SIZE solo es para mostrar las imágenes de prueba, si se quiere ver un batch de entrenamiento
# se debe asignar show_size al correspondiente del dataset,
# por ejemplo, CARVANA_BATCH_SIZE o ROAD_BATCH_SIZE
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
                    None,
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                "max"  # pooling
            ),
            (  # nconvs+concat: [nconvs, concat]
                [  # nconvs: [conv, conv, ...]
                    None,
                    (64, 3, "relu"),  # conv: [f, s, a]
                    (64, 3, "relu")
                ],
                True  # concat
            )
        ),
        (
            (
                [
                    None,
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    None,
                    (128, 3, "relu"),
                    (128, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    None,
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    None,
                    (256, 3, "relu"),
                    (256, 3, "relu")
                ],
                True
            )
        ),
        (
            (
                [
                    None,
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                "max"
            ),
            (
                [
                    None,
                    (512, 3, "relu"),
                    (512, 3, "relu")
                ],
                True
            )
        )
    ],
    [  # bottleneck: [conv, conv, ...]
        None,
        (1024, 3, "relu"),  # conv: [f, s, a]
        (1024, 3, "relu")
    ]
)

c = Chromosome(
    chromosome=unet_paper,
    max_layers=4,
    max_convs_per_layer=2
)

print("Name:\n", c)
print("Decoded:\n", c.get_decoded())
print("Real:\n", c.get_real())
print("Binary:\n", c.get_binary())
print("Binary:\n", c.get_binary(zip=True))

# Comprobamos que la decodificación y codificación sean correctas
assert c.get_decoded() == unet_paper
# Aunque tenga letras se trata de una codificación binaria
# Solo está comprimida para que no sea tan larga
assert (
    c.get_binary(zip=True) ==
    "AAAEIUIUABCFCGABRDRCQAMI4IYAGUOUKABVDVDAA6R6RIAHUPUMADSHSE_282"
)

c2 = Chromosome(chromosome=c.get_binary(zip=True))

print("\n\nName:\n", c2)
print("Decoded:\n", c2.get_decoded())
print("Real:\n", c2.get_real())
print("Binary:\n", c2.get_binary())
print("Binary:\n", c2.get_binary(zip=True))

assert c2.get_decoded() == unet_paper
assert c2.get_real() == c.get_real()
assert c2.get_binary() == c.get_binary()
assert c2.get_binary(zip=True) == c.get_binary(zip=True)

c3 = Chromosome(chromosome=c2.get_real())

print("\n\nName:\n", c3)
print("Decoded:\n", c3.get_decoded())
print("Real:\n", c3.get_real())
print("Binary:\n", c3.get_binary())
print("Binary:\n", c3.get_binary(zip=True))

assert c3.get_decoded() == unet_paper
assert c3.get_real() == c.get_real()
assert c3.get_binary() == c.get_binary()
assert c3.get_binary(zip=True) == c.get_binary(zip=True)


# %% Prueba de entrenamiento y evaluación
from codec import Chromosome

unet_paper = "AVCVCKRKRUIUISEKEPCHCFRDRD2R2RNI5I7UPUI_188"
c = Chromosome(
    max_layers=4,
    max_convs_per_layer=2,
    chromosome=unet_paper
)

# Mostramos la arquitectura a generar
print(c.get_decoded())

# data_loader = "carvana"
# data_path = "../data/carvana-dataset/"
# data_loader = "coco-people"
# data_path = "../data/coco-dataset-people/"
data_loader = "coco-car"
data_path = "../data/coco-dataset-car/"
# data_loader_alternativo = "car"
# data_path_alternativo = "../data/car-dataset/"
# data_loader_alternativo2 = "road"
# data_path_alternativo2 = "../data/road-dataset/"

# Al entrenarla se generará el modelo UNet automáticamente
c.train_unet(
    data_loader=data_loader,
    epochs=1,
    data_path=data_path,
    dataset_len=20_000
)

# Si no especificamos un DataLoader, se usará el dataloader con el que se entrenó
c.show_results()
# Aunque se haya entrenado con un DataLoader, podemos evaluar con otro
# c.show_results(data_loader_alternativo, data_path=data_path_alternativo)
# c.show_results(data_loader_alternativo2, data_path=data_path_alternativo2)

# Tampoco es necesario especificar el DataLoader si ya se ha entrenado
print("Aptitud para el DataLoader de entrenamiento (Loss):", c.get_aptitude())
# print("Aptitud para el otro DataLoader:", c.get_aptitude(
#     data_loader=data_loader_alternativo,
#     data_path=data_path_alternativo
# ))
# print("Aptitud para el 3er DataLoader:", c.get_aptitude(
#     data_loader=data_loader_alternativo2,
#     data_path=data_path_alternativo2
# ))

# # Guardamos los resultados
# c.show_results(
#     save=True
# )
# c.show_results(
#     data_loader_alternativo,
#     save=True,
#     name="para el otro DataLoader.png",
#     data_path=data_path_alternativo
# )
# c.show_results(
#     data_loader_alternativo2,
#     save=True,
#     name="para el otro DataLoader.png",
#     data_path=data_path_alternativo2
# )

# c.save_unet()

# %% Prueba de congruencia entre discretización y decodificación
FILTERS = {
    "0000": None,  # No aplicar convoluciones
    "0001": 1,
    "0011": 2,
    "0010": 4,
    "0110": 8,
    "0111": 16,
    "0101": 32,
    "0100": 64,
    "1100": 128,
    "1101": 256,
    "1111": 512,
    "1110": 1024,
}
KERNEL_SIZES = {
    "00": 1,
    "01": 3,
    "11": 5,
}
ACTIVATION_FUNCTIONS = {
    "0000": "linear",
    "0001": "relu",
    "0011": "softplus",
    "0010": "elu",
    "0110": "selu",
    "0111": "sigmoid",
    "0101": "tanh",
    "0100": "softsign",
    "1100": "softmax"
}
VALID_POOLINGS = {
    "01": "max",
    "11": "average",
}
POOLINGS = {
    "00": None  # No aplicar pooling
} | VALID_POOLINGS
CONCATENATION = {
    "0": False,
    "1": True,
}


def discretize_gene(value: int | bool | str, options: dict) -> str:
    step = 1 / len(options)
    real_rep = 0.01

    for _, v in options.items():
        if v == value:
            return round(real_rep, 2)

        real_rep += step

    raise ValueError(f"Value {value} not found in options")


def decode_gene(value: float, options: dict) -> int | bool | str:
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
from codec import Chromosome

unet_tmp = "OG2DGZF2WEDVMCTBSMEAO_104"
c = Chromosome(
    max_layers=4,
    max_convs_per_layer=2,
    chromosome=unet_tmp
)

# Mostramos la arquitectura a generar
print(c.get_decoded())


# %%
