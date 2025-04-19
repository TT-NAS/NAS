"""
Script para generar, entrenar y evaluar modelos de segmentación de imágenes
"""
import os
import math
from typing import Union, Optional

import pandas as pd
import matplotlib.pyplot as plt

from utils import RESULTS_PATH, IMAGES_PATH
from utils import OutOfMemoryError, TorchDataLoader
from utils import empty_cache_torch
from codec import Chromosome, MAX_LAYERS, MAX_CONVS_PER_LAYER


RESULTS_FILE = os.path.join(RESULTS_PATH, "results.csv")
LOG_FILE = os.path.join(RESULTS_PATH, "log.txt")


def plot_learning_curves(metrics: dict[str, list[float]], save: bool = False,
                         name: str = "learning curves.png", path: str = IMAGES_PATH):
    """
    Genera gráficas para visualizar la evolución de métricas de entrenamiento y validación

    Parameters
    ----------
    metrics : dict[str, list[float]]
        Diccionario donde cada llave es una métrica y cada lista es la evolución
        de esa métrica a lo largo de las épocas. Para cada llave `"train_{nombre_metrica}"`
        puede existir `"val_{nombre_metrica}"`
    save : bool, optional
        Si guardar la gráfica, by default `False`
    name : str, optional
        Nombre del archivo donde se guardará la gráfica, by default `"learning curves.png"`
    path : str, optional
        Ruta donde se guardará la gráfica, by default `IMAGES_PATH`
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
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        plt.savefig(os.path.join(path, name))
        print(
            f"-> Gráfico de curva de aprendizaje guardado en {os.path.join(path, name)}"
        )

        df = pd.DataFrame()

        len_data = len(metrics["train_loss"])
        epochs = range(1, len_data + 1)
        df["epoch"] = epochs

        for key, values in metrics.items():
            if not values:
                values = [None] * len_data

            df[key] = values

        name = os.path.splitext(name)[0] + ".csv"
        df.to_csv(os.path.join(path, name), index=False)
        print(
            f"-> Datos de curva de aprendizaje guardados en {os.path.join(path, name)}"
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
        Si se normalizan los valores de las métricas, by default `True`
    file : str, optional
        Archivo en el que se encuentran los resultados, by default `RESULTS_FILE`
    save : bool, optional
        Si guardar la gráfica, by default `False`
    name : str, optional
        Nombre del archivo donde se guardará la gráfica, by default `"scores.png"`
    path : str, optional
        Ruta donde se guardará la gráfica, by default `IMAGES_PATH`
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
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        plt.savefig(os.path.join(path, name))
        print(f"-> Gráfico de scores guardado en {os.path.join(path, name)}")
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
        Archivo en el que se registran los resultados, by default `RESULTS_FILE`
    """
    row = {
        "seed": chromosome.seed if chromosome.seed is not None else "N",
        "binary codification": chromosome.get_binary(zip=True),
        "max layers": chromosome.max_layers,
        "max convs per layer": chromosome.max_convs_per_layer,
        "layers": chromosome.get_num_layers(),
        "training secs": time_seconds,
        "epochs": last_epoch + 1
    }
    row.update(scores)

    if not os.path.exists(file):
        encabezado = list(row.keys())

        with open(file, "w") as f:
            f.write(",".join(encabezado) + "\n")

    df = pd.read_csv(file)
    df = df.dropna(axis=1, how='all')

    new_row = pd.DataFrame([row])
    new_row["seed"] = new_row["seed"].astype(object)

    if not new_row.dropna(how="all").empty:
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
        Archivo en el que se registra el mensaje, by default `LOG_FILE`
    """
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("")

    with open(file, "a") as f:
        f.write(message + "\n")


def score_model(dataset: str, chromosome: Optional[Union[tuple, list, str]] = None,
                seed: Optional[int] = None, alternative_datasets: Optional[list[str]] = None,
                save_pretrained_results: bool = False,
                **kwargs: Union[str, int, float, bool]) -> bool:
    """
    Obiene los puntajes de distintas métricas de un modelo

    Parameters
    ----------
    dataset : str
        Nombre del dataset a utilizar

        Opciones:
            - "coco-people"
            - "coco-car"
            - "carvana"
            - "road"
            - "car"
    chromosome : tuple or list or str, optional
        Cromosoma para asignar al modelo (hace que se ignore `seed`), by default `None`
    seed : int, optional
        Semilla para generar el cromosoma, by default `None`
    alternative_datasets : Optional[list], optional
        Lista de nombres de datasets con los que probar el modelo, además del
        dataset principal, by default `None`

        Opciones:
            - "coco-people"
            - "coco-car"
            - "carvana"
            - "road"
            - "car"
    save_pretrained_results : bool, optional
        Si entrenar una epoch del modelo y guardar los resultados, by default `False`
    **kwargs : T.Compose or str or int or float or bool
        Argumentos adicionales para la creación del cromosoma:
        - max_layers : (int) Máximo número de capas de la red sin contar el bottleneck
                       (`max_layers` no puede ser mayor que `codec.MAX_LAYERS`)
        - max_convs_per_layer : (int) Máximo número de capas convolucionales por bloque
                                (`max_convs_per_layer` no puede ser mayor que
                                `codec.MAX_CONVS_PER_LAYER`)

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

        Argumentos adicionales para el DataLoader:
        - batch_size : (int) Tamaño del batch
        - train_val_prop : (float) Proporción que se usará entre train y validation

        Argumentos adicionales para el Dataset:
        - data_path : (str) Ruta de los datos
        - length : (int) Número de imágenes a cargar
        - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                      entrenamiento (train y validation) y test
        - img_width : (int) Ancho a redimensionar las imágenes
        - img_height : (int) Alto a redimensionar las imágenes

    Returns
    -------
    bool
        Si el modelo se pudo entrenar correctamente
    """
    data_loader_args, kwargs = TorchDataLoader.get_args(kwargs)
    data_loader = TorchDataLoader(dataset, **data_loader_args)

    if alternative_datasets is None:
        alternative_datasets = []

    max_layers = kwargs.pop("max_layers", MAX_LAYERS)
    max_convs_per_layer = kwargs.pop(
        "max_convs_per_layer",
        MAX_CONVS_PER_LAYER
    )

    if chromosome:
        c = Chromosome(
            chromosome=chromosome,
            max_layers=max_layers,
            max_convs_per_layer=max_convs_per_layer
        )
    else:
        c = Chromosome(
            seed=seed,
            max_layers=max_layers,
            max_convs_per_layer=max_convs_per_layer
        )

    print("Modelo:", c.get_decoded())
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
        plot_learning_curves(
            metrics,
            save=True,
            name=f"learning curves {dataset} {last_epoch + 1}-epochs.png",
            path=save_path
        )

        c.save_unet()
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


def score_n_models(idx_start: Optional[int] = None, num: Optional[int] = None,
                   chromosomes: Optional[list[Union[tuple,
                                                    list[float], str]]] = None,
                   seeds: list[int] = None, dataset: str = "coco-car",
                   **kwargs: Union[str, int, float, bool]):
    """
    Obtiene los puntajes de distintas métricas de varios modelos

    Parameters
    ----------
    idx_start : Optional[int], optional
        Índice de inicio (si no se especifica `chromosomes` o `seeds`), by default `None`
    num : int, optional
        Cantidad de modelos a evaluar
        (si no se especifica `chromosomes` o `seeds`), by default `None`
    chromosomes : Optional[list[Union[tuple, list[float], str]]], optional
        Cromosomas para asignar a los modelos (hace que se ignore `seeds`), by default `None`
    seeds : list[int], optional
        Semillas para generar los cromosomas, by default `None`
    dataset : str, optional
        Nombre del dataset a utilizar, by default `"coco-car"`

        Opciones:
            - "coco-people"
            - "coco-car"
            - "carvana"
            - "road"
            - "car"
    **kwargs : str or int or float or bool
        Argumentos para la evaluación de cada modelo:
        - alternative_datasets : (list) Lista de nombres de datasets con los que probar el modelo,
                                 además del dataset principal ("coco-people", "coco-car",
                                 "carvana", "road", "car")
        - save_pretrained_results : (bool) Si entrenar una epoch del modelo y guardar los resultados

        Argumentos adicionales para la creación de los cromosomas:
        - max_layers : (int) Máximo número de capas de la red sin contar el bottleneck
                       (`max_layers` no puede ser mayor que `codec.MAX_LAYERS`)
        - max_convs_per_layer : (int) Máximo número de capas convolucionales por bloque
                                (`max_convs_per_layer` no puede ser mayor que
                                `codec.MAX_CONVS_PER_LAYER`)

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

        Argumentos adicionales para el DataLoader:
        - batch_size : (int) Tamaño del batch
        - train_val_prop : (float) Proporción que se usará entre train y validation

        Argumentos adicionales para el Dataset:
        - data_path : (str) Ruta de los datos
        - length : (int) Número de imágenes a cargar
        - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                      entrenamiento (train y validation) y test
        - img_width : (int) Ancho a redimensionar las imágenes
        - img_height : (int) Alto a redimensionar las imágenes
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
    score_n_models(
        idx_start=10,
        num=10,
        alternative_datasets=["carvana", "car"]
    )
