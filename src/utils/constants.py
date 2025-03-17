import os


# Número de elementos totales de entrenamiento de cada dataset
COCO_PEOPLE_DATASET_LENGTH = 64_115
COCO_CAR_DATASET_LENGTH = 12_251
CARVANA_DATASET_LENGTH = 5_088
ROAD_DATASET_LENGTH = 300
CAR_DATASET_LENGTH = 512

# Tamaño de los batches de cada dataset
COCO_BATCH_SIZE = 4
CARVANA_BATCH_SIZE = 4
ROAD_BATCH_SIZE = 1
CAR_BATCH_SIZE = 4
SHOW_SIZE = 32

# Rutas de los datasets
DATASETS_PATH = "./data"
COCO_PEOPLE_DATA_PATH = os.path.join(DATASETS_PATH, "coco-dataset-people")
COCO_CAR_DATA_PATH = os.path.join(DATASETS_PATH, "coco-dataset-car")
CARVANA_DATA_PATH = os.path.join(DATASETS_PATH, "carvana-dataset")
ROAD_DATA_PATH = os.path.join(DATASETS_PATH, "road-dataset")
CAR_DATA_PATH = os.path.join(DATASETS_PATH, "car-dataset")

# Ruta de los checkpoints de los modelos
CHECKPOINT_PATH = "./__checkpoints__"

# Rutas de los resultados
RESULTS_PATH = "./results"
IMAGES_PATH = os.path.join(RESULTS_PATH, "imgs")
MODELS_PATH = os.path.join(RESULTS_PATH, "models")

# Dimensiones de las imágenes
WIDTH = 256
HEIGHT = 256
CHANNELS = 3

# IDs de los datasets que usan COCO
COCO_IDS = {
    "people": "cpp",
    "car": "cca"
}
