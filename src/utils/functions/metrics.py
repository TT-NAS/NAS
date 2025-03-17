"""
Módulo con funciones para el cálculo de métricas de evaluación de modelos de segmentación

Funciones
---------
- dice_loss: Calcula la pérdida de Dice
- dice_crossentropy_loss: Calcula la pérdida de Dice-Crossentropy
- iou_loss: Calcula la pérdida de IoU
- accuracy: Calcula la precisión de la máscara predicha
"""
import torch
from torch import Tensor
from torch.nn import functional as F


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
