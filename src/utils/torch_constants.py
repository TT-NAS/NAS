import torch
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch._inductor.utils"
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
mp.set_start_method("spawn", force=True)

CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU")
