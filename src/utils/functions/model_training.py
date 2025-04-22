"""
Módulo con funciones para el entrenamiento y la evaluación de modelos UNet

Funciones
---------
- eval_model: Evalúa un modelo en base a una lista de métricas
- train_model: Entrena un modelo UNet con los datos de un DataLoader
"""
import math
from typing import Union

import torch
from torch import Tensor
from torch.amp import autocast, GradScaler
from colorama import Fore

from ..classes import UNet, TorchDataLoader
from ..globals import CUDA, LOGGER
from .checkpoint_manager import load_checkpoint, set_checkpoint
from .metrics import iou_loss, dice_loss, dice_crossentropy_loss, accuracy_loss


def eval_model(scores: Tensor, target: Tensor, metrics: list[str],
               clone: bool = True, loss: bool = True, items=False) -> Union[list[float], Tensor]:
    """
    Evalúa un modelo en base a una lista de métricas

    Parameters
    ----------
    scores : Tensor
        Resultado del modelo
    target : Tensor
        Máscara real
    metrics : list
        Métricas a evaluar
    clone : bool, optional
        Si se clonan los tensores, by default `True`
    loss : bool, optional
        Si se evalúa la pérdida, by default `False`
    items : bool, optional
        Si se devuelven los valores los items o los tensores, by default `False`

    Returns
    -------
    list or Tensor
        Lista de métricas si `loss=False`, Tensor con la pérdida si `loss=True`
    """
    METRICS = {
        "iou": iou_loss,
        "dice": dice_loss,
        "accuracy": accuracy_loss
    }
    results = []

    for metric in metrics:
        if clone:
            scores_tmp = scores.clone().to(CUDA)
            target_tmp = target.clone().to(CUDA)
        else:
            scores_tmp = scores
            target_tmp = target

        if metric == "dice crossentropy":
            result = dice_crossentropy_loss(scores_tmp, target_tmp)
        else:
            scores_tmp = torch.sigmoid(scores_tmp)
            pred_mask = scores_tmp[:, 0, ...]
            target_mask = target_tmp[:, 0, ...]
            result = METRICS[metric](pred_mask, target_mask)

        if loss:
            results.append(result)
        else:
            results.append(1 - result)

    if items:
        for i, result in enumerate(results):
            results[i] = result.item()

    return results


def train_model(model: UNet, data_loader: TorchDataLoader, metric: str = "iou", lr: float = 0.01,
                epochs: int = 15, early_stopping: bool = False, early_stopping_patience: int = 5,
                early_stopping_delta: float = 0.001, stopping_threshold: float = 0.05,
                infinite: bool = False, show_val: bool = True) -> tuple[UNet,
                                                                        int,
                                                                        dict[str, list[float]]]:
    """
    Entrena un modelo UNet

    Parameters
    ----------
    model : UNet
        Modelo a entrenar
    data_loader : TorchDataLoader
        DataLoader con los datos de entrenamiento y validación
    metric : str, optional
        Métrica a utilizar para calcular la pérdida, by default `"iou"`

        Opciones:
            - "iou"
            - "dice"
            - "dice crossentropy"
    lr : float, optional
        Tasa de aprendizaje, by default `0.01`
    epochs : int, optional
        Número de épocas, by default `15`
    early_stopping : bool, optional
        Si usar early stopping, by default `False`
    early_stopping_patience : int, optional
        Número de épocas a esperar sin mejora antes de detener el entrenamiento, by default `5`
    early_stopping_delta : float, optional
        Umbral mínimo de mejora para considerar un progreso, by default `0.001`
    stopping_threshold : float, optional
        Umbral de rendimiento para la métrica de validación. Si se alcanza o supera,
        el entrenamiento se detiene, by default `0.05`
    infinite : bool, optional
        Si el entrenamiento es infinito, by default `False`
    show_val : bool, optional
        Si mostrar los resultados de la validación en cada epoch, by default `True`

    Returns
    -------
    tuple
        (Modelo entrenado, última época, resultados de las métricas a lo largo del entrenamiento)
    """
    model = model.to(CUDA)
    model, metrics_results, initial_epoch = load_checkpoint(model)
    len_data = len(data_loader.train)
    len_val = len(data_loader.validation)

    best_loss = float("inf")
    counter = 0
    best_model_state = None

    scaler = GradScaler("cuda")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=1e-4
    )

    if infinite:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=max(math.ceil(early_stopping_patience / 2), 1),
            threshold=early_stopping_delta,
            min_lr=1e-6,
        )

        epochs = 100
        early_stopping = True
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-1,
            steps_per_epoch=len_data,
            epochs=epochs,
            pct_start=0.43,
            div_factor=10,
            final_div_factor=1000,
            three_phase=True
        )

    if early_stopping and not show_val:
        LOGGER.warning("Show_val asignado a True para early stopping")
        show_val = True

    train_loss_ant = torch.tensor(1.0)
    val_loss_ant = torch.tensor(1.0)

    if metrics_results is None:
        metrics_results = {
            "train_loss": [],
            "train_iou": [],
            "train_dice": [],
            "train_dice crossentropy": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_dice crossentropy": [],
            "val_accuracy": []
        }

    for epoch in range(initial_epoch, epochs):
        model.train()
        total_train_loss = 0
        total_train_iou = 0
        total_train_dice = 0
        total_train_dice_ce = 0
        total_train_acc = 0
        loss_ant = torch.tensor(1.0)
        print(f"=== Epoch [{epoch + 1}/{epochs}] ===")

        for i, (images, masks) in enumerate(data_loader.train):
            images = images.to(CUDA)
            masks = masks.to(CUDA)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                output = model(images)
                loss = eval_model(
                    scores=output,
                    target=masks,
                    metrics=[metric],
                    clone=False
                )[0]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not infinite:
                scheduler.step()

            with torch.no_grad():
                iou, dice, dice_ce, acc = eval_model(
                    scores=output.float(),
                    target=masks.float(),
                    metrics=["iou", "dice", "dice crossentropy", "accuracy"],
                    loss=False,
                    items=True
                )

            total_train_loss += loss.item()
            total_train_iou += iou
            total_train_dice += dice
            total_train_dice_ce += dice_ce
            total_train_acc += acc

            if i == 0 or (i + 1) % 5 == 0 or i + 1 == len_data:
                evol_loss = Fore.GREEN + "↓" if loss < loss_ant else Fore.RED + "↑"
                evol_loss += Fore.RESET
                loss_ant = loss

                print(
                    f"\rBatch [{i + 1}/{len_data}], Loss: {loss.item():.4f}, "
                    f"IoU: {iou:.4f}, Dice: {dice:.4f}, "
                    f"Dice CE: {dice_ce:.4f}, Acc: {acc:.4f}, "
                    f"Evolución Loss: {evol_loss}",
                    end=" "
                )

        evol_train_loss = (
            Fore.GREEN + "↓"
            if total_train_loss / len_data < train_loss_ant
            else Fore.RED + "↑"
        )
        evol_train_loss += Fore.RESET
        train_loss_ant = total_train_loss
        print(
            "\n-- "
            f"Loss: {total_train_loss / len_data:.4f}, "
            f"IoU: {total_train_iou / len_data:.4f}, "
            f"Dice: {total_train_dice / len_data:.4f}, "
            f"Dice CE: {total_train_dice_ce / len_data:.4f}, "
            f"Acc: {total_train_acc / len_data:.4f}, "
            f"Evolución Train Loss: {evol_train_loss}"
            " --"
        )
        metrics_results["train_loss"].append(total_train_loss / len_data)
        metrics_results["train_iou"].append(total_train_iou / len_data)
        metrics_results["train_dice"].append(total_train_dice / len_data)
        metrics_results["train_dice crossentropy"].append(
            total_train_dice_ce / len_data
        )
        metrics_results["train_accuracy"].append(total_train_acc / len_data)

        if not early_stopping:
            set_checkpoint(model.state_dict(), metrics_results, epoch)

        # Validación y early stopping
        if not show_val:
            if total_train_loss / len_data < best_loss:
                best_loss = total_train_loss / len_data
                best_model_state = model.state_dict().copy()

            continue

        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        total_val_dice = 0
        total_val_dice_ce = 0
        total_val_acc = 0

        with torch.no_grad():
            for images, masks in data_loader.validation:
                images = images.to(CUDA)
                masks = masks.to(CUDA)

                with autocast(device_type="cuda", dtype=torch.float16):
                    output = model(images)
                    loss = eval_model(
                        scores=output.float(),
                        target=masks.float(),
                        metrics=[metric],
                        items=True
                    )[0]

                iou, dice, dice_ce, acc = eval_model(
                    scores=output.float(),
                    target=masks.float(),
                    metrics=[
                        "iou", "dice",
                        "dice crossentropy", "accuracy"
                    ],
                    loss=False,
                    items=True
                )

                total_val_loss += loss
                total_val_iou += iou
                total_val_dice += dice
                total_val_dice_ce += dice_ce
                total_val_acc += acc

        avg_val_loss = total_val_loss / len_val
        avg_val_iou = total_val_iou / len_val
        avg_val_dice = total_val_dice / len_val
        avg_val_dice_ce = total_val_dice_ce / len_val
        avg_val_acc = total_val_acc / len_val
        evol_val_loss = (
            Fore.GREEN + "↓"
            if avg_val_loss < val_loss_ant
            else Fore.RED + "↑"
        )
        evol_val_loss += Fore.RESET
        val_loss_ant = avg_val_loss
        print(
            "--- "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val IoU: {avg_val_iou:.4f}, "
            f"Val Dice: {avg_val_dice:.4f}, "
            f"Val Dice CE: {avg_val_dice_ce:.4f}, "
            f"Val Acc: {avg_val_acc:.4f}, "
            f"Evolución Val Loss: {evol_val_loss}"
            " ---"
        )
        metrics_results["val_loss"].append(avg_val_loss)
        metrics_results["val_iou"].append(avg_val_iou)
        metrics_results["val_dice"].append(avg_val_dice)
        metrics_results["val_dice crossentropy"].append(avg_val_dice_ce)
        metrics_results["val_accuracy"].append(avg_val_acc)

        if not early_stopping:
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = model.state_dict().copy()

            continue

        if infinite:
            scheduler.step(avg_val_loss)

        if avg_val_loss < stopping_threshold:
            print(
                f"¡Umbral de rendimiento alcanzado! {metric}: "
                f"{avg_val_loss:.4f} < {stopping_threshold:.4f}"
            )
            best_model_state = model.state_dict().copy()
            break

        if epoch == 99:
            print("¡Límite de épocas alcanzado!")
            break

        LOGGER.info(
            "Comparación de métricas para early stopping: "
            f"{avg_val_loss < best_loss - early_stopping_delta=}"
        )

        if avg_val_loss < best_loss - early_stopping_delta:
            best_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            LOGGER.info(
                f"Sin mejora en la pérdida. Contador: {counter}/{early_stopping_patience}"
            )

        if counter >= early_stopping_patience:
            print(
                "Early stopping activado por falta de mejora. "
                f"La mejor pérdida fue: {best_loss:.4f}"
            )
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

        print()

        set_checkpoint(best_model_state, metrics_results, epoch)

    # Asegurarse de que el modelo final sea el mejor encontrado
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Se ha restaurado el mejor modelo encontrado.")

    return model, epoch, metrics_results
