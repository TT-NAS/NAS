"""
Script para generar, entrenar y evaluar modelos de segmentación de imágenes
"""
import os
import math

import pandas as pd
from typing import Union, Optional
import matplotlib.pyplot as plt

from utils import RESULTS_PATH, IMAGES_PATH
from utils import OutOfMemoryError, TorchDataLoader
from utils import empty_cache_torch
from codec import Chromosome


RESULTS_FILE = os.path.join(RESULTS_PATH, "results.csv")
LOG_FILE = os.path.join(RESULTS_PATH, "log.txt")


def plot_learning_curve(metrics: dict[str, list[float]], save: bool = False,
                        name: str = "learning curve.png", path: str = IMAGES_PATH):
    """
    Genera gráficas para visualizar la evolución de métricas de entrenamiento y validación.

    Parameters:
    -----------
    metrics : dict[str, list[float]]
        Diccionario donde cada llave es una métrica y cada lista es la evolución
        de esa métrica a lo largo de las épocas.
        Para cada llave "train_{nombre_metrica}" puede existir "val_{nombre_metrica}".

    Returns:
    --------
    None, muestra las gráficas generadas.
    """
    train_metrics = {}
    val_metrics = {}
    base_metric_names = set()

    for key, values in metrics.items():
        if key.startswith("train_"):
            base_name = key[6:]  # Quitar "train_"
            train_metrics[base_name] = values
            base_metric_names.add(base_name)
        elif key.startswith("val_"):
            base_name = key[4:]  # Quitar "val_"
            val_metrics[base_name] = values
            base_metric_names.add(base_name)

    # Verificar si hay métricas de validación no vacías
    has_val_metrics = any(len(values) > 0 for values in val_metrics.values())

    # Calcular número de subplots: una para cada métrica + 2 para todas las train y val
    num_subplots = len(base_metric_names)
    if has_val_metrics:
        num_subplots += 2  # Agregar plots para "all_train" y "all_val"
    else:
        num_subplots += 1  # Solo agregar plot para "all_train"

    # Calcular dimensiones de la gráfica
    cols = math.ceil(math.sqrt(num_subplots))
    rows = math.ceil(num_subplots / cols)

    _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_subplots > 1 else [axes]

    colors = ["red", "green", "blue", "orange", "purple",
              "brown", "pink", "gray", "olive", "cyan"]

    plot_idx = 0

    for base_name in sorted(base_metric_names):
        ax = axes[plot_idx]

        # Graficar train
        if base_name in train_metrics:
            train_values = train_metrics[base_name]
            epochs = range(1, len(train_values) + 1)
            ax.plot(epochs, train_values, 'b-', label=f'Train {base_name}')

        # Graficar val (si existe)
        if base_name in val_metrics and len(val_metrics[base_name]) > 0:
            val_values = val_metrics[base_name]
            epochs = range(1, len(val_values) + 1)
            ax.plot(epochs, val_values, 'r-', label=f'Val {base_name}')

        ax.set_title(f'{base_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

        plot_idx += 1

    # Graficar todas las métricas de entrenamiento juntas
    ax = axes[plot_idx]
    for i, base_name in enumerate(sorted(base_metric_names)):
        if base_name in train_metrics:
            train_values = train_metrics[base_name]
            epochs = range(1, len(train_values) + 1)
            color_idx = i % len(colors)
            ax.plot(epochs, train_values,
                    color=colors[color_idx], label=f'Train {base_name}')

    ax.set_title('All Training Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    plot_idx += 1

    # Graficar todas las métricas de validación juntas (si existen)
    if has_val_metrics:
        ax = axes[plot_idx]
        for i, base_name in enumerate(sorted(base_metric_names)):
            if base_name in val_metrics and len(val_metrics[base_name]) > 0:
                val_values = val_metrics[base_name]
                epochs = range(1, len(val_values) + 1)
                color_idx = i % len(colors)
                ax.plot(epochs, val_values,
                        color=colors[color_idx], label=f'Val {base_name}')

        ax.set_title('All Validation Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

        plot_idx += 1

    # Ocultar ejes vacíos si hay más subplots que métricas
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path, name))
        print(
            f"Gráfico de curva de aprendizaje guardada en {os.path.join(path, name)}"
        )
    else:
        plt.show()

    plt.close()


def plot_scores_and_metrics(selected_columns: list[str], normalize: bool = True,
                            file: str = RESULTS_FILE, save: bool = False, name: str = "scores.png",
                            path: str = IMAGES_PATH):
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
            "Las columnas seleccionadas no son válidas."
        )

    remaining_columns = [col for col in df.columns[6:]
                         if col not in selected_columns]
    values = {key: df[key].to_list() for key in df.columns[6:]}

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

    colors = [
        "red", "green", "blue", "orange", "purple",
        "brown", "pink", "gray", "olive", "cyan"
    ]

    for i, col in enumerate(remaining_columns):
        sorted_indices = sorted(
            range(len(values[col])),
            key=lambda k: values[col][k],
            reverse=col == "loss"
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

    # Ocultar ejes vacíos si hay más subplots que métricas
    for i in range(num_subplots, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path, name))
        print(f"Gráfico de scores guardada en {os.path.join(path, name)}")
    else:
        plt.show()

    plt.close()


def reg_results(chromosome: Chromosome, time_seconds: float, last_epoch: int,
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
        Archivo en el que se registran los resultados, by default RESULTS_FILE
    """
    df = pd.read_csv(file)
    row = {
        "seed": chromosome.seed if chromosome.seed is not None else "N",
        "binary codification": chromosome.get_binary(zip=True),
        "max convs per layer": chromosome.max_convs_per_layer,
        "layers": chromosome.get_num_layers(),
        "training secs": time_seconds,
        "epochs": last_epoch + 1
    }

    row.update(scores)

    new_row = pd.DataFrame([row])
    new_row["seed"] = new_row["seed"].astype(object)
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


def score_model(dataset: str, chromosome: Optional[Union[tuple, list, str]] = None,
                seed: Optional[int] = None, max_layers: int = 3, max_convs_per_layer: int = 2,
                alternative_datasets: Optional[list[str]] = None,
                save_pretrained_results: bool = True,
                **kwargs: Union[str, int, float, bool, object]) -> bool:
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
    alternative_datasets : Optional[list], optional
        Lista de nombres de datasets con los que probar el modelo, además del
        dataset principal, by default []
    save_pretrained_results : bool, optional
        Si entrenar una epoch del modelo y guardar los resultados, by default True
    **kwargs : T.Compose or str or int or float or bool
        Argumentos adicionales para el entrenamiento:
        - metric : (str) Métrica a utilizar para calcular la pérdida
                   ("iou", "dice" o "dice crossentropy")
        - lr : (float) Tasa de aprendizaje
        - epochs : (int) Número de épocas
        - early_stopping_patience : (int) Número de épocas a esperar sin mejora antes de detener
                                    el entrenamiento
        - early_stopping_delta : (float) Umbral mínimo de mejora para considerar un progreso
        - stopping_threshold : (float) Umbral de rendimiento para la métrica de validación. Si se
                               alcanza o supera, el entrenamiento se detiene
        - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
        - print_every : (int) Cada cuántos pasos se imprime el resultado

        Argumentos adicionales para el DataLoader:
        - batch_size : (int) Tamaño del batch
        - train_val_prop : (float) Proporción que se usará entre train y validation

        Argumentos adicionales para el dataset:
        - data_path : (str) Ruta de los datos
        - length : (int) Número de imágenes a cargar
        - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                      entrenamiento (train y validation) y test
        - transform : (T.Compose) Transformaciones a aplicar a las imágenes

    Returns
    -------
    bool
        Si el modelo se pudo entrenar correctamente
    """
    data_loader_args, kwargs = TorchDataLoader.get_args(kwargs)
    data_loader = TorchDataLoader(dataset, **data_loader_args)

    if alternative_datasets is None:
        alternative_datasets = []

    if chromosome:
        c = Chromosome(
            max_layers=max_layers,
            max_convs_per_layer=max_convs_per_layer,
            chromosome=chromosome
        )
    else:
        c = Chromosome(
            max_layers=max_layers,
            max_convs_per_layer=max_convs_per_layer,
            seed=seed
        )

    save_path = os.path.join(IMAGES_PATH, str(c))

    try:
        # Métricas
        if save_pretrained_results:
            kwargs2 = kwargs.copy()
            kwargs2["epochs"] = 1
            c.train_unet(data_loader, **kwargs2)
            c.show_results(
                save=True,
                path=save_path,
                name=f"results {dataset} 1-epoch.png"
            )

            for alt_dataset in alternative_datasets:
                alt_dataloader = TorchDataLoader(
                    alt_dataset,
                    **data_loader_args,
                )
                c.data_loader = str(alt_dataloader)
                c.show_results(
                    data_loader=alt_dataloader,
                    save=True,
                    path=save_path,
                    name=f"results {alt_dataset} 1-epoch.png"
                )

        c.set_unet()
        empty_cache_torch()
        time_seconds, last_epoch, metrics = c.train_unet(data_loader, **kwargs)

        # si se realizó validación en cada epoch
        if metrics["val_loss"]:
            scores_dict = {
                "loss": metrics["val_loss"][-1],
                "iou": metrics["val_iou"][-1],
                "dice": metrics["val_dice"][-1],
                "dice crossentropy": metrics["val_dice crossentropy"][-1],
                "accuracy": metrics["val_accuracy"][-1]
            }
        else:
            scores_dict = {
                "loss": metrics["train_loss"][-1],
                "iou": metrics["train_iou"][-1],
                "dice": metrics["train_dice"][-1],
                "dice crossentropy": metrics["train_dice crossentropy"][-1],
                "accuracy": metrics["train_accuracy"][-1]
            }

        reg_results(
            chromosome=c,
            time_seconds=time_seconds,
            last_epoch=last_epoch,
            scores=scores_dict
        )

        c.save_unet()
        plot_learning_curve(
            metrics,
            save=True,
            name=f"learning curve {dataset} {last_epoch + 1}-epochs.png",
            path=save_path
        )
        c.show_results(
            save=True,
            path=save_path,
            name=f"results {dataset} {last_epoch + 1}-epochs.png"
        )

        if not alternative_datasets:
            return True

        for alt_dataset in alternative_datasets:
            alt_dataloader = TorchDataLoader(
                alt_dataset,
                **data_loader_args,
            )
            c.data_loader = str(alt_dataloader)
            c.show_results(
                data_loader=alt_dataloader,
                save=True,
                path=save_path,
                name=f"results {alt_dataset} {last_epoch + 1}-epochs.png"
            )

        return True
    except OutOfMemoryError:
        log("ERROR:")
        log("  + Semilla: " + str(seed))
        log("  + Binary cod: " + c.get_binary(zip=True))
        log("  - Error: CUDA se quedó sin memoria")
    except KeyboardInterrupt:
        print("El entrenamiento fue interrumpido")
        exit()
    except Exception as e:
        log("ERROR:")
        log("  + Semilla: " + str(seed))
        log("  + Binary cod: " + c.get_binary(zip=True))
        log("  - Error:" + str(e))
    finally:
        empty_cache_torch()

    return False


def score_n_models(idx_start: int = None, num: int = None,
                   chromosomes: Optional[list[Union[tuple,
                                                    list[float], str]]] = None,
                   seeds: list[int] = None, dataset: str = "carvana",
                   **kwargs: Union[str, int, float, bool]):
    """
    Obtiene los puntajes de distintas métricas de varios modelos

    Parameters
    ----------
    start : int, optional
        Índice de inicio (si no se especifica `chromosomes` o `seeds`), by default None
    num : int, optional
        Cantidad de modelos a evaluar
        (si no se especifica `chromosomes` o `seeds`), by default None
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
        - alternative_datasets : (list) Lista de nombres de datasets con los que probar el modelo,
                                 además del dataset principal

        Argumentos adicionales para el entrenamiento:
        - metric : (str) Métrica a utilizar para calcular la pérdida
                   ("iou", "dice" o "dice crossentropy")
        - lr : (float) Tasa de aprendizaje
        - epochs : (int) Número de épocas
        - early_stopping_patience : (int) Número de épocas a esperar sin mejora antes de detener el
                                    entrenamiento
        - early_stopping_delta : (float) Umbral mínimo de mejora para considerar un progreso
        - stopping_threshold : (float) Umbral de rendimiento para la métrica de validación. Si se
                               alcanza o supera, el entrenamiento se detiene
        - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
        - print_every : (int) Cada cuántos pasos se imprime el resultado

        Argumentos adicionales para el DataLoader:
        - batch_size : (int) Tamaño del batch
        - train_val_prop : (float) Proporción que se usará entre train y validation

        Argumentos adicionales para el dataset:
        - data_path : (str) Ruta de los datos
        - length : (int) Número de imágenes a cargar
        - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                      entrenamiento (train y validation) y test
        - transform : (T.Compose) Transformaciones a aplicar a las imágenes
    """
    if chromosomes:
        for c in chromosomes:
            score_model(dataset, chromosome=c, **kwargs)
    elif seeds:
        for s in seeds:
            score_model(dataset, seed=s, **kwargs)
    elif idx_start is not None and num is not None:
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
                print("Se han intentado entrenar demasiados modelos sin éxito seguidos")
                log("Se han intentado entrenar demasiados modelos sin éxito seguidos")
                break


if __name__ == "__main__":
    # UNet Paper
    # score_n_models(
    #     chromosomes=["AVCVCKRKRUIUISEKEPCHCFRDRD2R2RNI5I7UPUI_188"],
    #     dataset_len=1000,
    #     alternative_datasets=["car"],
    #     max_layers=4
    # )
    score_n_models(
        idx_start=0,
        num=23,
        dataset_len=1000,
        alternative_datasets=["car"],
    )
    plot_scores_and_metrics(
        selected_columns=["synflow", "gradient"],
        save=True
    )
