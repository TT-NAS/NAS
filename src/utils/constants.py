import os
import logging


CARVANA_DATASET_LENGTH = 5088
ROAD_DATASET_LENGTH = 300
CAR_DATASET_LENGTH = 512

CARVANA_BATCH_SIZE = 4
ROAD_BATCH_SIZE = 1
CAR_BATCH_SIZE = 4
SHOW_SIZE = 32

DATASETS_PATH = "./data"
CARVANA_DATA_PATH = os.path.join(DATASETS_PATH, "carvana-dataset")
ROAD_DATA_PATH = os.path.join(DATASETS_PATH, "road-dataset")
CAR_DATA_PATH = os.path.join(DATASETS_PATH, "car-dataset")

RESULTS_PATH = "./results_ant"
IMAGES_PATH = os.path.join(RESULTS_PATH, "imgs")
MODELS_PATH = os.path.join(RESULTS_PATH, "models")

WIDTH = 224
HEIGHT = 224
CHANNELS = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

LOGGER = logging.getLogger()
