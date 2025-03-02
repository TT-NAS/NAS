import os
from typing import Union

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
    WIDTH, HEIGHT, CHANNELS,
    CUDA
)


def plot_batch(imgs: Tensor, masks: Tensor, save: bool = False, show_size: int = SHOW_SIZE, name: str = "batch.png", path: str = IMAGES_PATH):
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
        Ruta donde se guardarán las imágenes, by default IMGS_PATH
    """
    # Mostrar las dimensiones de los un batch de imágenes y máscaras
    # print(f" - Imágenes: {imgs.shape}")
    # print(f" - Máscaras: {masks.shape}")
    # Mostrar un batch de imágenes y máscaras
    plt.figure(figsize=(20, 10))

    for i in range(show_size):
        plt.subplot(4, 8, i + 1)
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


def dice_loss(pred_mask: Tensor, target_mask: Tensor, smooth: float = 1e-8, **_) -> Tensor:
    """
    Calcula la pérdida de Dice

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real
    smooth : float, optional
        Valor para suavizar la división, by default 1e-8

    Returns
    -------
    Tensor
        Pérdida de Dice
    """
    target_mask = target_mask.float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice


def dice_crossentropy_loss(scores: Tensor, target: Tensor, **kwargs: float) -> Tensor:
    """
    Calcula la pérdida de Dice-Crossentropy

    Parameters
    ----------
    scores : Tensor
        Resultado del modelo
    target : Tensor
        Máscara real
    smooth : float, optional
        Valor para suavizar la división, by default 1e-8
    **kwargs : float
        Argumentos adicionales para la función `dice_loss`:
        - smooth : (float) Valor para suavizar la división

    Returns
    -------
    Tensor
        Pérdida de Dice-Crossentropy
    """
    ce = F.binary_cross_entropy_with_logits(scores, target)

    scores = torch.sigmoid(scores)
    pred_mask = scores[:, 0, ...]
    target_mask = target[:, 0, ...]
    dice = dice_loss(pred_mask, target_mask, **kwargs)

    return (ce + dice) / 2


def iou_loss(pred_mask: Tensor, target_mask: Tensor, smooth: float = 1e-8, **_) -> Tensor:
    """
    Calcula la pérdida de IoU

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real
    smooth : float, optional
        Valor para suavizar la división, by default 1e-8

    Returns
    -------
    Tensor
        Pérdida de IoU
    """
    target_mask = target_mask.float()

    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask - pred_mask * target_mask).sum()
    iou = (intersection + smooth) / (union + smooth)

    return 1 - iou


def accuracy(pred_mask: Tensor, target_mask: Tensor, threshold: float = 0.5, **_) -> Tensor:
    """
    Calcula la precisión de la máscara predicha

    Parameters
    ----------
    pred_mask : Tensor
        Máscara predicha
    target_mask : Tensor
        Máscara real
    threshold : float, optional
        Umbral para considerar si un píxel es 1 o 0, by default

    Returns
    -------
    Tensor
        Precisión de la máscara predicha
    """
    pred_mask = (pred_mask > threshold).float()
    target_mask = target_mask.float()

    correct = (pred_mask == target_mask).float()
    acc = correct.mean()

    return 1 - acc


def eval_model(scores: Tensor, target: Tensor, metrics: list[str], loss: bool = True, items=False, **kwargs: float) -> Union[list[float], Tensor]:
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
    loss : bool, optional
        Si se evalúa la pérdida, by default False
    items : bool, optional
        Si se devuelven los valores los items o los tensores, by default False
    **kwargs : float
        Argumentos adicionales para las funciones de pérdida:
        - smooth : (float) Valor para suavizar la división
        - threshold : (float) Umbral para considerar si un píxel es 1 o 0

    Returns
    -------
    list or Tensor
        Lista de métricas si `loss=False`, Tensor con la pérdida si `loss=True`
    """
    METRICS = {
        "iou": iou_loss,
        "dice": dice_loss,
        "accuracy": accuracy
    }
    results = []
    scores = scores.to(CUDA)
    target = target.to(CUDA)

    for metric in metrics:
        if metric == "dice crossentropy":
            result = dice_crossentropy_loss(scores, target, **kwargs)
        else:
            scores = torch.sigmoid(scores)
            pred_mask = scores[:, 0, ...]
            target_mask = target[:, 0, ...]
            result = METRICS[metric](pred_mask, target_mask, **kwargs)

        if loss:
            results.append(result)
        else:
            results.append(1 - result)

    if items:
        for i, result in enumerate(results):
            results[i] = result.item()

    return results


def train_model(model: UNet, data_loader: TorchDataLoader, metric: str = "iou", lr: float = 0.01, epochs: int = 5, show_val: bool = False, print_every: int = 25):
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
            - "accuracy"
    lr : float, optional
        Tasa de aprendizaje, by default 0.01
    epochs : int, optional
        Número de épocas, by default 5
    show_val : bool, optional
        Si mostrar los resultados de la validación en cada epoch, by default False
    print_every : int, optional
        Cada cuántos pasos se imprime el resultado, by default 25
    """
    len_data = len(data_loader.train)
    model = model.to(CUDA)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=1e-4
    )
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

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        total_train_dice = 0
        total_train_dice_ce = 0
        total_train_iou = 0
        print(f"=== Epoch [{epoch + 1}/{epochs}] ===")

        for i, (images, masks) in enumerate(data_loader.train):
            images = images.to(CUDA, dtype=torch.float32)
            masks = masks.to(CUDA, dtype=torch.float32)

            output = model(images)
            loss = eval_model(
                scores=output,
                target=masks,
                metrics=[metric]
            )[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            with torch.no_grad():
                acc, dice, dice_ce, iou = eval_model(
                    scores=output,
                    target=masks,
                    metrics=["accuracy", "dice", "dice crossentropy", "iou"],
                    loss=False,
                    items=True
                )

            total_train_loss += loss
            total_train_acc += acc
            total_train_dice += dice
            total_train_dice_ce += dice_ce
            total_train_iou += iou

            if i == 0 or (i + 1) % print_every == 0 or i + 1 == len_data:
                print(
                    f"Batch [{i + 1}/{len_data}], Loss: {loss:.4f}, "
                    f"Acc: {acc:.4f}, Dice: {dice:.4f}, "
                    f"Dice CE: {dice_ce:.4f}, IoU: {iou:.4f}"
                )

        print(
            f"Loss: {total_train_loss / len_data:.4f}, "
            f"Acc: {total_train_acc / len_data:.4f}, "
            f"Dice: {total_train_dice / len_data:.4f}, "
            f"Dice CE: {total_train_dice_ce / len_data:.4f}, "
            f"IoU: {total_train_iou / len_data:.4f}"
        )

        if show_val:
            len_val = len(data_loader.validation)
            model.eval()
            total_val_loss = 0
            total_val_acc = 0
            total_val_dice = 0
            total_val_dice_ce = 0
            total_val_iou = 0

            with torch.no_grad():
                for images, masks in data_loader.validation:
                    images = images.to(CUDA)
                    masks = masks.to(CUDA)

                    output = model(images)

                    loss, acc, dice, dice_ce, iou = eval_model(
                        scores=output,
                        target=masks,
                        metrics=[
                            metric, "accuracy", "dice",
                            "dice crossentropy", "iou"
                        ],
                        loss=False,
                        items=True
                    )

                    total_val_loss += loss
                    total_val_acc += acc
                    total_val_dice += dice
                    total_val_dice_ce += dice_ce
                    total_val_iou += iou

            print(
                f"Val Loss: {total_val_loss / len_val:.4f}, "
                f"Val Acc: {total_val_acc / len_val:.4f}, "
                f"Val Dice: {total_val_dice / len_val:.4f}, "
                f"Val Dice CE: {total_val_dice_ce / len_val:.4f}, "
                f"Val IoU: {total_val_iou / len_val:.4f}",
                end="\n\n"
            )


def save_model(model: nn.Module, name: str, path: str = MODELS_PATH):
    """
    Guarda un modelo en un archivo

    Parameters
    ----------
    model : nn.Module
        Modelo a guardar
    name : str
        Nombre del archivo
    path : str, optional
        Ruta donde se guardará el modelo, by default MODELS_PATH
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, name))


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


def gradient_scorer_pytorch_ant(model, batch_size: int = 32):
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
    # Set the hooks
    def gradient_hook(module, grad_input, grad_output):
        gradients.append((
            module.__class__.__name__,
            grad_output[0].detach().clone()
        ))

    # Asignar el hook a todas las capas convolucionales
    def assign_hook(module):
        for _, child in module.named_children():
            if len(list(child.children())):
                assign_hook(child)
            elif isinstance(child, nn.Conv2d):
                child.register_full_backward_hook(gradient_hook)

    # Genera entradas aleatorias en el formato correcto (batch_size, channels, height, width)

    X_train = torch.randn((batch_size, CHANNELS, HEIGHT, WIDTH))
    # Mapa de etiquetas (sin canal extra)
    Y_train = torch.randn((batch_size, HEIGHT, WIDTH)).unsqueeze(1)

    # Imprimir los tipos de las entradas
    # print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    # print(f"X_train: {X_train.dtype}, Y_train: {Y_train.dtype}")

    # Compute the scores
    scores = []
    gradients = []

    # Definir la función de pérdida
    criterion = F.binary_cross_entropy_with_logits

    # Forward
    # Debe retornar (batch_size, num_classes, height, width)
    output = model(X_train)
    loss = criterion(output, Y_train)  # Calcula la pérdida

    model.zero_grad()
    loss.backward()

    # Process
    addition_back = []
    for i, (name, grad) in enumerate(gradients):
        # Average of the magnitude of the gradients
        addition_grad = grad.detach().clone().nan_to_num(
            nan=0).abs().mean(dim=0).flatten()
        # Min-Max normalization
        addition_grad = (addition_grad - addition_grad.min()) / \
            (addition_grad.max() - addition_grad.min())
        addition_back.append(addition_grad.mean())
    addition_back = torch.tensor(addition_back).nan_to_num(0)
    score = torch.sum(addition_back).log().item()
    return score
