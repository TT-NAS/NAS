from codec import Chromosome
from base_architectures.vgg import VGG_FCN
from utils.classes import TorchDataLoader
from utils import train_model
from base_architectures.unet import unet_paper
from multiprocessing import freeze_support
from utils import plot_results
import pandas as pd

import os
import time

RESULTS_FILE = os.path.join("./results/base_architectures", "results.csv")


def reg_results(architecture_name: str, time_seconds: float, last_epoch: int,
                scores: dict[str, float], file: str = RESULTS_FILE):
    """
    Registra los resultados de un modelo en un archivo CSV

    Parameters
    ----------
    chromosome : Chromosome
        Cromosoma del modelo
    time_seconds : float
        Tiempo de entrenamiento en segundos
    last_epoch : int
        Última época en la que se entrenó
    scores : dict
        Puntajes de las métricas
    file : str, optional
        Archivo en el que se registran los resultados, by default `RESULTS_FILE`
    """
    row = {
        "architecture": architecture_name,
        "training secs": time_seconds,
        "epochs": last_epoch + 1
    }
    row.update(scores)

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
    
def main():

    data_loader = TorchDataLoader("road", batch_size=4, train_val_prop=0.8)
    #versions = ["32", "16", "8"]
    versions = ["8"]
    max_epochs = 15
    
    # Entrenamiento FCN-VGG
    for version in versions:
        fcn_model = VGG_FCN(version=version)
        
        start = time.perf_counter()
        model, last_epoch, metrics = train_model(fcn_model, data_loader, epochs=max_epochs, lr=1e-5)
        time_seconds = time.perf_counter() - start
        
        results_name = f"fcn_vgg_{fcn_model.version}_results.png"
        results_path = "./results/base_architectures"
        plot_results(model, data_loader.test, save=True, name=results_name, path=results_path)
        
        if metrics["val_loss"]:
            scores_dict = {
                "val_loss": metrics["val_loss"][-1],
                "val_iou": metrics["val_iou"][-1],
                "val_dice": metrics["val_dice"][-1],
                "val_dice crossentropy": metrics["val_dice crossentropy"][-1],
                "val_accuracy": metrics["val_accuracy"][-1]
            }
        else:
            scores_dict = {
                "val_loss": None,
                "val_iou": None,
                "val_dice": None,
                "val_dice crossentropy": None,
                "val_accuracy": None
            }

        scores_dict.update({
            "train_loss": metrics["train_loss"][-1],
            "train_iou": metrics["train_iou"][-1],
            "train_dice": metrics["train_dice"][-1],
            "train_dice crossentropy": metrics["train_dice crossentropy"][-1],
            "train_accuracy": metrics["train_accuracy"][-1]
        })

        reg_results(
            architecture_name=f"fcn_vgg_{fcn_model.version}",
            time_seconds=time_seconds,
            last_epoch=last_epoch,
            scores=scores_dict
        )
        
    # Entrenamiento U-Net
    c = Chromosome(chromosome=unet_paper)
    model = c.get_unet()
    
    start = time.perf_counter()
    model, last_epoch, metrics = train_model(model, data_loader, epochs=max_epochs, lr=1e-2)
    time_seconds = time.perf_counter() - start
    
    results_name = "unet_results.png"
    results_path = "./results/base_architectures"
    plot_results(model, data_loader.test, save=True, name=results_name, path=results_path)
    
    reg_results(
        architecture_name="unet",
        time_seconds=time_seconds,
        last_epoch=last_epoch,
        scores={
            "val_loss": metrics["val_loss"][-1],
            "val_iou": metrics["val_iou"][-1],
            "val_dice": metrics["val_dice"][-1],
            "val_dice crossentropy": metrics["val_dice crossentropy"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
            "train_loss": metrics["train_loss"][-1],
            "train_iou": metrics["train_iou"][-1],
            "train_dice": metrics["train_dice"][-1],
            "train_dice crossentropy": metrics["train_dice crossentropy"][-1],
            "train_accuracy": metrics["train_accuracy"][-1]
        }
    )
    
if __name__ == "__main__":
    freeze_support()
    main()
    