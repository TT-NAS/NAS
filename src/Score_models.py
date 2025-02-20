"""
Script para generar, entrenar y evaluar modelos de segmentación de imágenes
"""
import torch
import pandas as pd
from typing import Union, Optional
import matplotlib.pyplot as plt

from Torch_utils import TorchDataLoader, gradient_scorer_pytorch, synflow, eval_model, CUDA
from Codec import Chromosome


METRICS_TO_EVAL = ["accuracy", "dice", "dice crossentropy", "iou"]


def plot_results(normalize=True, file="results.csv"):
    """
    Grafica los resultados de los modelos entrenados

    Parameters
    ----------
    normalize : bool, optional
        Si se normalizan los valores de las métricas, by default True
    file : str, optional
        Archivo en el que se encuentran los resultados, by default "results.csv"
    """
    df = pd.read_csv(file)
    df = df.sort_values(by="iou")
    values = {
        key: df[key].to_list()
        for key in df.columns[1:]
    }

    if normalize:
        values = {
            key: [
                (v - min(values[key])) / (max(values[key]) - min(values[key]))
                for v in values[key]
            ]
            for key in values.keys()
        }

    # graficar
    _, ax = plt.subplots()
    colors = ["red", "green", "blue", "orange", "purple"]
    index = 0

    for key in values.keys():
        color = colors[index % len(colors)]
        ax.plot(values[key], label=key, color=color)
        index += 1

    ax.legend()
    plt.show()


def reg_results(scores: dict[str, float], name: str, file="results.csv"):
    """
    Registra los resultados de un modelo en un archivo CSV

    Parameters
    ----------
    syn : float
        Puntaje de synflow
    scores : dict
        Puntajes de las métricas
    name : str
        Nombre del modelo
    file : str, optional
        Archivo en el que se registran los resultados, by default "results.csv"
    """
    df = pd.read_csv(file)
    row = {
        "id": name
    }

    row.update(scores)

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)


def score_model(dataset: str, chromosome: Optional[Union[tuple, list, str]] = None, seed: Optional[int] = None, epochs: int = 1, max_layers: int = 3, max_conv_per_layer: int = 2) -> bool:
    """
    Obiene los puntajes de distintas métricas de un modelo

    Parameters
    ----------
    dataset : str
        Nombre del dataset a utilizar

        Opciones:
            - "carvana"
            - "road"
    chromosome : tuple or list or str, optional
        Cromosoma para asignar al modelo (hace que se ignore `seed`), by default None
    seed : int, optional
        Semilla para generar el cromosoma, by default None
    epochs : int, optional
        Número de épocas para entrenar el modelo, by default 1
    max_layers : int, optional
        Máximo número de capas para el modelo, by default 3
    max_conv_per_layer : int, optional
        Máximo número de convoluciones por capa, by default 2

    Returns
    -------
    bool
        Si el modelo se pudo entrenar correctamente
    """
    data_loader = TorchDataLoader(dataset)

    if chromosome:
        c = Chromosome(
            max_layers=max_layers,
            max_conv_per_layer=max_conv_per_layer,
            chromosome=chromosome
        )
    else:
        c = Chromosome(
            max_layers=max_layers,
            max_conv_per_layer=max_conv_per_layer,
            seed=seed
        )

    # Predictores
    # jaime = gradient_scorer_pytorch(c.get_unet())
    syn = synflow(c.get_unet())

    # Métricas
    try:
        c.train_unet(data_loader, epochs=epochs)
    except torch.OutOfMemoryError:
        print("ERROR:")
        print("  + Semilla: " + str(seed) if seed else "Semilla: None")
        print("  + Binary cod:", c.get_binary(zip=True))
        print("  - Error: CUDA se quedó sin memoria")

        return False
    except Exception as e:
        print("ERROR:")
        print("  + Semilla: " + str(seed) if seed else "Semilla: None")
        print("  + Binary cod:", c.get_binary(zip=True))
        print("  - Error:", e)

        return False

    model = c.get_unet()
    model.eval()
    imgs, masks = next(iter(data_loader.validation))
    imgs = imgs.to(CUDA)
    model = model.to(CUDA)
    outputs = model(imgs)

    with torch.no_grad():
        scores = eval_model(
            scores=outputs,
            target=masks,
            metrics=METRICS_TO_EVAL,
            loss=False,
            items=True
        )

    scores_dict = {
        "synflow": syn,
        # "gradient": jaime
    }

    for metric, score in zip(METRICS_TO_EVAL, scores):
        scores_dict[metric] = score

    if seed:
        name = f"{seed}_" + c.get_binary(zip=True)
    else:
        name = c.get_binary(zip=True)

    reg_results(scores_dict, name)
    c.show_results(save=True, name=name)
    c.save_unet(name + ".pt")

    return True


def score_n_models(idx_start: int = None, num: int = None, chromosomes: Optional[list[Union[tuple, list[float], str]]] = None, seeds: list[int] = None, dataset: str = "carvana", **kwargs: int):
    """
    Obtiene los puntajes de distintas métricas de varios modelos

    Parameters
    ----------
    start : int, optional
        Índice de inicio (si no se especifica `chromosomes` o `seeds`), by default None
    num : int, optional
        Cantidad de modelos a evaluar (si no se especifica `chromosomes` o `seeds`), by default None
    chromosomes : Optional[list[Union[tuple, list[float], str]]], optional
        Cromosomas para asignar a los modelos (hace que se ignore `seeds`), by default None
    seeds : list[int], optional
        Semillas para generar los cromosomas, by default None
    dataset : str, optional
        Nombre del dataset a utilizar, by default "carvana"

        Opciones:
            - "carvana"
            - "road"
    **kwargs : int
        Argumentos para `score_model`:
        - epochs : (int) Número de épocas para entrenar el modelo
        - max_layers : (int) Máximo número de capas para el modelo
        - max_conv_per_layer : (int) Máximo número de convoluciones por capa
    """
    if chromosomes:
        for c in chromosomes:
            score_model(dataset, chromosome=c, **kwargs)
    elif seeds:
        for s in seeds:
            score_model(dataset, seed=s, **kwargs)
    elif idx_start and num:
        j = idx_start
        i = idx_start

        while j < idx_start + num:
            succes = score_model(dataset, seed=i, **kwargs)

            if succes:
                j += 1

            i += 1


if __name__ == "__main__":
    score_n_models(idx_start=21, num=1, dataset="road", epochs=2)
    plot_results()
