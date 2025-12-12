# Entrenamiento de las arquitecturas SOTA con validación cruzada
import os
import time

from multiprocessing import freeze_support
import pandas as pd
from sklearn.model_selection import KFold

# Monkey patch para deshabilitar checkpoints
from utils.functions import checkpoint_manager
def load_checkpoint(model, path):
    return model, None, 0
def set_checkpoint(model_state, metrics_results, current_epoch, path):
    return
checkpoint_manager.load_checkpoint = load_checkpoint
checkpoint_manager.set_checkpoint = set_checkpoint

from base_architectures.unet import unet_paper
from base_architectures.vgg import VGG_FCN
from codec import Chromosome
from utils.constants import ROAD_DATASET_LENGTH
from utils.classes import TorchDataLoader
from utils.functions import set_current_net_binary
from utils import train_model, plot_results

RESULTS_PATH = os.path.join("./results/base_architectures/k_folds")

def reg_results(architecture_name: str, fold: int, time_seconds: float, last_epoch: int,
                metrics, path: str = RESULTS_PATH):
    """
    Registra los resultados del entrenamiento en un archivo CSV

    Parameters
    ----------
    time_seconds : float
        Tiempo de entrenamiento en segundos
    last_epoch : int
        Última época en la que se entrenó
    scores : dict
        Puntajes de las métricas
    file : str, optional
        Archivo en el que se registran los resultados, by default `RESULTS_FILE`
    """
    scores_dict = {
            "val_loss": metrics["val_loss"][-1],
            "val_iou": metrics["val_iou"][-1],
            "val_dice": metrics["val_dice"][-1],
            "val_dice crossentropy": metrics["val_dice crossentropy"][-1],
            "val_accuracy": metrics["val_accuracy"][-1]
        }
    scores_dict.update({
        "train_loss": metrics["train_loss"][-1],
        "train_iou": metrics["train_iou"][-1],
        "train_dice": metrics["train_dice"][-1],
        "train_dice crossentropy": metrics["train_dice crossentropy"][-1],
        "train_accuracy": metrics["train_accuracy"][-1]
    })
    row = {
        "fold": fold,
        "training secs": time_seconds,
        "epochs": last_epoch + 1
    }
    row.update(scores_dict)
    
    file = os.path.join(path, f"{architecture_name}.csv")

    if not os.path.exists(file):
        encabezado = list(row.keys())

        with open(file, "w") as f:
            f.write(",".join(encabezado) + "\n")

    df = pd.read_csv(file)
    df = df.dropna(axis=1, how="all")

    new_row = pd.DataFrame([row])

    if not new_row.dropna(how="all").empty:
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)

def k_fold_idxs(dataset_len: int, n_splits: int = 5, random_state: int = 42):
    """
    Genera los índices para k-folds

    Parameters
    ----------
    n_batches : int
        Longitud del dataset
    n_splits : int, optional
        Número de folds, by default `5`
    random_state : int, optional
        Semilla para la aleatoriedad, by default `42`

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Lista de tuplas con los índices de train y validation para cada fold
    """
    k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(k_folds.split(range(dataset_len)))

def main():
    dataset_len = int(ROAD_DATASET_LENGTH * 0.8) # Evitar los índices del 0.2 de test
    k_folds = k_fold_idxs(dataset_len , n_splits=5, random_state=42)
    
    versions = []
    max_epochs = 10
    images_path = "./results/base_architectures/k_folds/imgs"
    
    # Entrenamiento FCN-VGG
    for version in versions:
        # Folds
        for fold, (train_idx, val_idx) in enumerate(k_folds):
            data_loader = TorchDataLoader("road", k_fold_idxs=(train_idx, val_idx))
            fcn_model = VGG_FCN(version=version)
            set_current_net_binary('')
            start = time.perf_counter()
            model, last_epoch, metrics = train_model(fcn_model, data_loader, epochs=max_epochs, lr=1e-5)
            time_seconds = time.perf_counter() - start
            
            results_name = f"fcn_vgg_{fcn_model.version}_{fold}.png"
            plot_results(model, data_loader.test, save=True, name=results_name, path=images_path, umbralize=False)
            reg_results(
                architecture_name=f"fcn_vgg_{fcn_model.version}",
                fold=fold,
                metrics=metrics,
                time_seconds=time_seconds,
                last_epoch=last_epoch
            )
            
    # Entrenamiento U-Net
    for fold, (train_idx, val_idx) in enumerate(k_folds):
        data_loader = TorchDataLoader("road", k_fold_idxs=(train_idx, val_idx))
        c = Chromosome(chromosome=unet_paper)
        model = c.get_unet()
        start = time.perf_counter()
        set_current_net_binary('')
        model, last_epoch, metrics = train_model(model, data_loader, epochs=max_epochs, lr=1e-2)
        time_seconds = time.perf_counter() - start
        
        results_name = f"unet_results_{fold}.png"
        
        plot_results(model, data_loader.test, save=True, name=results_name, path=images_path)
        
        reg_results(
            architecture_name="unet",
            fold=fold,
            metrics=metrics,
            time_seconds=time_seconds,
            last_epoch=last_epoch,
        )
        
if __name__ == "__main__":
    freeze_support()
    main()
    