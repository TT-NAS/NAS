import os
import math
from typing import Union, Optional

# Torch
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .classes import UNet, TorchDataLoader
from .constants import (
    SHOW_SIZE,
    MODELS_PATH, IMAGES_PATH,
    CUDA,
    LOGGER
)


def plot_batch(imgs: Tensor, masks: Tensor, save: bool = False, show_size: int = SHOW_SIZE,
               name: str = "batch.png", path: str = IMAGES_PATH):
    """
    Muestra un conjunto de imágenes y máscaras junto con sus dimensiones

    Parameters
    ----------
    imgs : Tensor
        Imágenes
    masks : Tensor
        Máscaras
    save : bool, optional
        Si se guardan las imágenes o se muestran, by default False
    show_size : int, optional
        Número de imágenes a mostrar, by default SHOW_SIZE
    name : str, optional
        Nombre del archivo a guardar, by default "batch.png"
    path : str, optional
        Ruta donde se guardarán las imágenes, by default IMAGES_PATH
    """
    # Mostrar un batch de imágenes y máscaras
    cols = math.ceil(math.sqrt(show_size))
    rows = math.ceil(show_size / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(show_size):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i, ...].permute(1, 2, 0).numpy()
        mask = masks[i, ...].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.axis("Off")

    plt.tight_layout()

    if save:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        plt.savefig(os.path.join(path, name))

        print(
            f"Gráfico de prueba con el batch guardado en {os.path.join(path, name)}"
        )
    else:
        plt.show()

    plt.close()


def plot_results(model: UNet, test_loader: DataLoader, **kwargs: Union[bool, str]):
    """
    Muestra los resultados de un modelo en un conjunto de imágenes

    Parameters
    ----------
    model : UNet
        Modelo a evaluar
    test_loader : DataLoader
        DataLoader con las imágenes a evaluar
    **kwargs : bool or str
        Argumentos adicionales para la función `plot_batch`:
        - save : (bool) Si se guardan las imágenes o se muestran
        - show_size : (int) Número de imágenes a mostrar
        - name : (str) Nombre del archivo a guardar
        - path : (str) Ruta donde se guardarán las imágenes
    """
    imgs = next(iter(test_loader))
    imgs = imgs.to(CUDA)
    model = model.to(CUDA)

    with torch.no_grad():
        model.eval()
        scores = model(imgs)
        result = (scores > 0.5).float()

    imgs = imgs.cpu()
    result = result.cpu()
    plot_batch(
        imgs=imgs,
        masks=result,
        **kwargs
    )


def dice_loss(pred_mask: Tensor, target_mask: Tensor) -> Tensor:
    """
    Calcula la pérdida de Dice

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real

    Returns
    -------
    Tensor
        Pérdida de Dice
    """
    smooth = 1e-8
    target_mask = target_mask.float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice


def dice_crossentropy_loss(scores: Tensor, target: Tensor) -> Tensor:
    """
    Calcula la pérdida de Dice-Crossentropy

    Parameters
    ----------
    scores : Tensor
        Resultado del modelo
    target : Tensor
        Máscara real

    Returns
    -------
    Tensor
        Pérdida de Dice-Crossentropy
    """
    ce = F.binary_cross_entropy_with_logits(scores, target)
    ce = torch.sigmoid(ce)

    scores = torch.sigmoid(scores)
    pred_mask = scores[:, 0, ...]
    target_mask = target[:, 0, ...]
    dice = dice_loss(pred_mask, target_mask)

    return (ce + dice) / 2


def iou_loss(pred_mask: Tensor, target_mask: Tensor) -> Tensor:
    """
    Calcula la pérdida de IoU

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real

    Returns
    -------
    Tensor
        Pérdida de IoU
    """
    smooth = 1e-8
    target_mask = target_mask.float()

    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask - pred_mask * target_mask).sum()
    iou = (intersection + smooth) / (union + smooth)

    return 1 - iou


def accuracy_loss(pred_mask: Tensor, target_mask: Tensor) -> Tensor:
    """
    Calcula la precisión de la máscara predicha

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real

    Returns
    -------
    Tensor
        Precisión de la máscara predicha
    """
    pred_mask = (pred_mask > 0.5).float()
    target_mask = target_mask.float()

    correct = (pred_mask == target_mask).float()
    acc = correct.mean()

    return 1 - acc


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
        Si se clonan los tensores, by default True
    loss : bool, optional
        Si se evalúa la pérdida, by default False
    items : bool, optional
        Si se devuelven los valores los items o los tensores, by default False

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
                epochs: Optional[int] = None, early_stopping_patience: int = 5,
                early_stopping_delta: float = 0.001, stopping_threshold: float = 0.05,
                show_val: bool = False, print_every: int = 25) -> tuple[UNet,
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
        Métrica a utilizar para calcular la pérdida, by default "iou"

        Opciones:
            - "iou"
            - "dice"
            - "dice crossentropy"
    lr : float, optional
        Tasa de aprendizaje, by default 0.01
    epochs : Optional[int], optional
        Número de épocas, si no se especifica, se activa el early stopping, by default None
    early_stopping_patience : int, optional
        Número de épocas a esperar sin mejora antes de detener el entrenamiento, by default 5
    early_stopping_delta : float, optional
        Umbral mínimo de mejora para considerar un progreso, by default 0.001
    stopping_threshold : float, optional
        Umbral de rendimiento para la métrica de validación. Si se alcanza o supera,
        el entrenamiento se detiene, by default 0.05
    show_val : bool, optional
        Si mostrar los resultados de la validación en cada epoch, by default False
    print_every : int, optional
        Cada cuántos pasos se imprime el resultado, by default 25

    Returns
    -------
    tuple
        (Modelo entrenado, última época, resultados de las métricas a lo largo del entrenamiento)
    """
    len_data = len(data_loader.train)
    len_val = len(data_loader.validation)
    model = model.to(CUDA)
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    early_stopping = epochs is None
    epochs = epochs if not early_stopping else 100

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=1e-4
    )

    if early_stopping:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=max(math.ceil(early_stopping_patience / 2), 1),
            threshold=early_stopping_delta,
            min_lr=1e-6,
        )

        if not show_val:
            LOGGER.warning("Show_val asignado a True para early stopping")
            show_val = True
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

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_iou = 0
        total_train_dice = 0
        total_train_dice_ce = 0
        total_train_acc = 0
        print(f"=== Epoch [{epoch + 1}/{epochs}] ===")

        for i, (images, masks) in enumerate(data_loader.train):
            images = images.to(CUDA)
            masks = masks.to(CUDA)

            output = model(images)
            loss = eval_model(
                scores=output,
                target=masks,
                metrics=[metric],
                clone=False
            )[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not early_stopping:
                scheduler.step()

            with torch.no_grad():
                iou, dice, dice_ce, acc = eval_model(
                    scores=output,
                    target=masks,
                    metrics=["iou", "dice", "dice crossentropy", "accuracy"],
                    loss=False,
                    items=True
                )

            total_train_loss += loss.item()
            total_train_iou += iou
            total_train_dice += dice
            total_train_dice_ce += dice_ce
            total_train_acc += acc

            if i == 0 or (i + 1) % print_every == 0 or i + 1 == len_data:
                print(
                    f"Batch [{i + 1}/{len_data}], Loss: {loss.item():.4f}, "
                    f"IoU: {iou:.4f}, Dice: {dice:.4f}, "
                    f"Dice CE: {dice_ce:.4f}, Acc: {acc:.4f}"
                )

        print(
            "- "
            f"Loss: {total_train_loss / len_data:.4f}, "
            f"IoU: {total_train_iou / len_data:.4f}, "
            f"Dice: {total_train_dice / len_data:.4f}, "
            f"Dice CE: {total_train_dice_ce / len_data:.4f}, "
            f"Acc: {total_train_acc / len_data:.4f}"
            " -"
        )
        metrics_results["train_loss"].append(total_train_loss / len_data)
        metrics_results["train_iou"].append(total_train_iou / len_data)
        metrics_results["train_dice"].append(total_train_dice / len_data)
        metrics_results["train_dice crossentropy"].append(
            total_train_dice_ce / len_data)
        metrics_results["train_accuracy"].append(total_train_acc / len_data)

        # Validación y early stopping
        if not show_val:
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

                output = model(images)

                loss = eval_model(
                    scores=output,
                    target=masks,
                    metrics=[metric],
                    items=True
                )[0]

                iou, dice, dice_ce, acc = eval_model(
                    scores=output,
                    target=masks,
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
        print(
            "--- "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val IoU: {avg_val_iou:.4f}, "
            f"Val Dice: {avg_val_dice:.4f}, "
            f"Val Dice CE: {avg_val_dice_ce:.4f}, "
            f"Val Acc: {avg_val_acc:.4f}"
            " ---"
        )
        metrics_results["val_loss"].append(avg_val_loss)
        metrics_results["val_iou"].append(avg_val_iou)
        metrics_results["val_dice"].append(avg_val_dice)
        metrics_results["val_dice crossentropy"].append(avg_val_dice_ce)
        metrics_results["val_accuracy"].append(avg_val_acc)

        if not early_stopping:
            continue

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
            f"({avg_val_loss} < {best_val_loss - early_stopping_delta}) = "
            f"{avg_val_loss < best_val_loss - early_stopping_delta}"
        )

        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            LOGGER.info(
                f"Sin mejora en la métrica {metric}. Contador: {counter}/{early_stopping_patience}"
            )

        if counter >= early_stopping_patience:
            print(
                "Early stopping activado por falta de mejora. "
                f"La mejor métrica {metric} fue: {best_val_loss:.4f}"
            )
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

        print()

    # Asegurarse de que el modelo final sea el mejor encontrado
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Se ha restaurado el mejor modelo encontrado.")

    return model, epoch, metrics_results


def save_model(model: UNet, name: str, path: str = MODELS_PATH):
    """
    Guarda un modelo en un archivo

    Parameters
    ----------
    model : UNet
        Modelo a guardar
    name : str
        Nombre del archivo
    path : str, optional
        Ruta donde se guardará el modelo, by default MODELS_PATH
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, name))

    print(f"Modelo guardado en {os.path.join(path, name)}")


def gradient_scorer_pytorch(model, batch_size: int = 4):
    """
    Score a model using the gradient activity of the network.

    Parameters
    ----------
    model : torch.nn.Module
        Model to score.
    batch_size : int, optional
        Size of the batches. The default is 5.

    Returns
    -------
    score : float
        Score of the model.
    """
    model = model.to(CUDA)
    # Set the hooks

    def gradient_hook(module, grad_input, grad_output):
        gradients.append((module.__class__.__name__,
                         grad_output[0].detach().clone()))

    # Asignar el hook a todas las capas convolucionales
    def assign_hook(module):
        for _, child in module.named_children():
            if len(list(child.children())):
                assign_hook(child)
            elif isinstance(child, nn.Conv2d):
                child.register_backward_hook(gradient_hook)

    assign_hook(model)

    data_loader = TorchDataLoader("road", batch_size=batch_size)
    train_dataloader = data_loader.train

    # Compute the scores
    scores = []
    gradients = []

    # Definir la función de pérdida
    criterion = F.binary_cross_entropy_with_logits

    # Forward
    # for image, mask in train_dataloader:
    image, mask = next(iter(train_dataloader))
    image = image.to(CUDA)
    mask = mask.to(CUDA)
    # Debe retornar (batch_size, num_classes, height, width)
    output = model(image)
    loss = criterion(output, mask)  # Calcula la pérdida
    model.zero_grad()
    loss.backward()

    # Process
    addition_back = []
    for i, (name, grad) in enumerate(gradients):
        # Average of the magnitude of the gradients
        addition_grad = grad.detach().clone().nan_to_num(nan=0).abs().sum(dim=0).flatten()
        # Min-Max normalization
        addition_grad = (addition_grad - addition_grad.min()) / \
            (addition_grad.max() - addition_grad.min())
        addition_back.append(addition_grad.sum(dim=0))
    addition_back = torch.tensor(addition_back).nan_to_num(0)
    score = torch.sum(addition_back).item()
    return score
