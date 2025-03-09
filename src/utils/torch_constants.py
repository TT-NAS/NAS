import torch
from torchvision import transforms as T

from .constants import WIDTH, HEIGHT


TRANSFORM = T.Compose([
    T.Resize([WIDTH, HEIGHT]),
    T.ToTensor()
])

CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU")
