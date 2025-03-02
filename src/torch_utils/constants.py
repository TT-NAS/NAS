import torch
from torchvision import transforms as T

CARVANA_BATCH_SIZE = 32
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

if __name__ == "__main__":
    xd = {"transform": TRANSFORM, "data_path": CARVANA_DATA_PATH}
    transform = repr(xd.get("transform", "0")).replace("\n", "").replace(" ", "").replace("'", "")
    print(transform)

    xd = {"data_path": CARVANA_DATA_PATH}
    transform = repr(xd.get("transform", "0")).replace("\n", "").replace(" ", "").replace("'", "")
    print(transform)

    xd = {"data_path": CARVANA_DATA_PATH}
    transform = xd.get("data_path", "0").replace("\n", "")
    print(transform)

    xd = {}
    transform = xd.get("data_path", "0").replace("\n", "")
    print(transform)
