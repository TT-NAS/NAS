import torch
import logging

from torchvision import transforms as T


CARVANA_DATASET_LENGTH = 5088
ROAD_DATASET_LENGTH = 300
CAR_DATASET_LENGTH = 512

CARVANA_BATCH_SIZE = 4
ROAD_BATCH_SIZE = 1
CAR_BATCH_SIZE = 4
SHOW_SIZE = 32

CARVANA_DATA_PATH = "./carvana-dataset/"
ROAD_DATA_PATH = "./road-dataset/"
CAR_DATA_PATH = "./car-dataset/"

MODELS_PATH = "./models/"
IMAGES_PATH = "./imgs/"

WIDTH = 224
HEIGHT = 224
CHANNELS = 3


TRANSFORM = T.Compose([
    T.Resize([WIDTH, HEIGHT]),
    T.ToTensor()
])

CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU")

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

LOGGER = logging.getLogger()
