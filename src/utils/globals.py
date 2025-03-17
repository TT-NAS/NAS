import torch
import logging
import warnings


# Configuraciones de PyTorch
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch._inductor.utils"
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU")

# Configuraciones del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Variables globales
CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger()
CURRENT_NET_BINARY = str()
