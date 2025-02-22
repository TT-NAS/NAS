"""
Script para generar, entrenar y evaluar modelos de segmentación de imágenes
"""
import math

import torch
import pandas as pd
from typing import Union, Optional
import matplotlib.pyplot as plt

from Torch_utils import TorchDataLoader, gradient_scorer_pytorch, eval_model, CUDA, Synflow
from Codec import Chromosome


METRICS_TO_EVAL = ["accuracy", "dice", "dice crossentropy", "iou"]
RESULTS_FILE = "results.csv"
LOG_FILE = "log.txt"


def plot_results(selected_columns: list[str], normalize: bool = True, file: str = RESULTS_FILE):
    """
    Grafica los resultados de los modelos entrenados en múltiples subgráficos.

    Parameters
    ----------
    selected_columns : list
        Lista de nombres de columnas a incluir en todas las gráficas.
    normalize : bool, optional
        Si se normalizan los valores de las métricas, by default True
    file : str, optional
        Archivo en el que se encuentran los resultados, by default RESULTS_FILE
    """
    df = pd.read_csv(file)

    if not all(col in df.columns for col in selected_columns):
        raise ValueError(
            "Las columnas seleccionadas no son válidas o no están en el DataFrame."
        )

    remaining_columns = [col for col in df.columns[1:]
                         if col not in selected_columns]
    values = {key: df[key].to_list() for key in df.columns[1:]}

    if normalize:
        if any(len(values[key]) <= 1 for key in values.keys()):
            return
        values = {
            key: [(v - min(values[key])) / (max(values[key]) - min(values[key]))
                  for v in values[key]]
            for key in values.keys()
        }

    # Medidas de los subplots
    num_subplots = len(remaining_columns)
    cols = math.ceil(math.sqrt(num_subplots))
    rows = math.ceil(num_subplots / cols)

    _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_subplots > 1 else [axes]

    colors = ["red", "green", "blue", "orange", "purple"]

    for i, col in enumerate(remaining_columns):
        sorted_indices = sorted(
            range(len(values[col])), key=lambda k: values[col][k]
        )
        ax = axes[i]

        for index, key in enumerate(selected_columns + [col]):
            color = colors[index % len(colors)]
            sorted_values = [values[key][j] for j in sorted_indices]
            ax.plot(sorted_values, label=key, color=color)

        ax.set_title(f"Comparación con {col}")
        ax.legend()

    for ax in axes:
        ax.set_ylim(-0.5, 1.5)

    plt.tight_layout()
    plt.show()


def reg_results(scores: dict[str, float], name: str, file=RESULTS_FILE):
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
        Archivo en el que se registran los resultados, by default RESULTS_FILE
    """
    df = pd.read_csv(file)
    row = {
        "id": name
    }

    row.update(scores)

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)


def log(message: str, file: str = LOG_FILE):
    """
    Registra un mensaje en un archivo de texto

    Parameters
    ----------
    message : str
        Mensaje a registrar
    file : str, optional
        Archivo en el que se registra el mensaje, by default LOG_FILE
    """
    with open(file, "a") as f:
        f.write(message + "\n")


def score_model(dataset: str, chromosome: Optional[Union[tuple, list, str]] = None, seed: Optional[int] = None, max_layers: int = 3, max_conv_per_layer: int = 2, **kwargs: Union[str, int, float, bool]) -> bool:
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
    max_layers : int, optional
        Máximo número de capas para el modelo, by default 3
    max_conv_per_layer : int, optional
        Máximo número de convoluciones por capa, by default 2
    **kwargs : str or int or float or bool
        Argumentos adicionales para el entrenamiento:
        - metric : (str) Métrica a utilizar para calcular la pérdida. ("iou", "dice", "dice crossentropy" o "accuracy")
        - lr : (float) Tasa de aprendizaje
        - epochs : (int) Número de épocas
        - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
        - print_every : (int) Cada cuántos pasos se imprime el resultado

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

    try:
        # Predictores
        synflow_scorer = Synflow(c.get_unet())
        syn = synflow_scorer.score(c.get_unet(), shape=[3, 224, 224])
        # syn + 1 para evitar log(0), prácticamente no afecta
        syn = math.log(syn + 1)

        # reseteamos la unet para evitar problemas con el gradiente
        c.set_unet()
        jaime = gradient_scorer_pytorch(c.get_unet())
        jaime = math.log(jaime + 1)

        # Métricas
        c.set_unet()
        c.train_unet(data_loader, **kwargs)

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
            "gradient": jaime
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
    except torch.OutOfMemoryError:
        log("ERROR:")
        log("  + Semilla: " + str(seed) if seed else "Semilla: None")
        log("  + Binary cod: " + c.get_binary(zip=True))
        log("  - Error: CUDA se quedó sin memoria")
    except KeyboardInterrupt:
        print("El entrenamiento fue interrumpido")
        exit()
    except Exception as e:
        log("ERROR:")
        log("  + Semilla: " + str(seed) if seed else "Semilla: None")
        log("  + Binary cod: " + c.get_binary(zip=True))
        log("  - Error:" + str(e))
    finally:
        torch.cuda.empty_cache()

    return False


def score_n_models(idx_start: int = None, num: int = None, chromosomes: Optional[list[Union[tuple, list[float], str]]] = None, seeds: list[int] = None, dataset: str = "carvana", **kwargs: Union[str, int, float, bool]):
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
    **kwargs : str or int or float or bool
        Argumentos para la función `score_model`:
        - max_layers : (int) Máximo número de capas para el modelo
        - max_conv_per_layer : (int) Máximo número de convoluciones por capa

        Argumentos adicionales para el entrenamiento:
        - metric : (str) Métrica a utilizar para calcular la pérdida. ("iou", "dice", "dice crossentropy" o "accuracy")
        - lr : (float) Tasa de aprendizaje
        - epochs : (int) Número de épocas
        - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
        - print_every : (int) Cada cuántos pasos se imprime el resultado
    """
    if chromosomes:
        for c in chromosomes:
            score_model(dataset, chromosome=c, **kwargs)
    elif seeds:
        for s in seeds:
            score_model(dataset, seed=s, **kwargs)
    elif idx_start and num:
        i = idx_start
        last_succes = idx_start
        seed = idx_start

        while i < idx_start + num:
            succes = score_model(dataset, seed=seed, **kwargs)

            if succes:
                i += 1
                last_succes = seed

            seed += 1

            if last_succes + 20 < seed:
                log("Se han intentado entrenar demasiados modelos sin éxito seguidos")
                break


if __name__ == "__main__":
    score_n_models(
        idx_start=139,
        num=2,
        dataset="road",
        epochs=10
    )
    plot_results(selected_columns=["synflow", "gradient"])
