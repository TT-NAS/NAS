import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn, Tensor
import torch
from torch.amp import autocast
from typing import *
from tqdm import tqdm

from codec import Chromosome
from utils import TorchDataLoader
from utils.globals import CUDA
from utils.functions.metrics import iou_loss, dice_loss, dice_crossentropy_loss, accuracy_loss


# Cargar los datos
df = pd.read_csv(f'/home/wimbo/NAS/results/results.csv')


def cosine_similarity_extended(tensors: Tensor) -> Tensor:
    # Dividir cada tensor entre su norma
    for tensor in tensors:
        tensor /= torch.norm(tensor) + 1e-8
    
    # Obtener el promedio de los tensores
    mean_tensor = torch.mean(tensors, dim=0, keepdim=False)
    
    # Norma cuadrada del tensor promedio
    mean_tensor_norm = torch.norm(mean_tensor)
    
    return mean_tensor_norm**2
    
# Funciones auxiliares
def eval_model(scores: Tensor, target: Tensor, metrics: list[str],
               clone: bool = True, loss: bool = True, items=False) -> Union[list[float], Tensor]:
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

# Intersection over union
def iou(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    smooth = 1e-8
    intersection = (tensor1 * tensor2).sum().numpy()
    union = (tensor1.sum() + tensor2.sum()) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def cosine_similarity(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm_a = torch.norm(tensor1)
    norm_b = torch.norm(tensor2)
    cos_sim = dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
    return cos_sim

# Función para obtener el tape de gradientes
def get_tape(model: nn.Module, data_loader: TorchDataLoader):
    # Modelo en modo entrenamiento
    model.train()
    model.to(CUDA, dtype=torch.float16)
    
    # Hook para guardar los gradientes
    gradient_tape = []
    gradient_tape_abs = []
    handles = []  # Para guardar los handles de los hooks
    
    def get_gradients(module, grad_input, grad_output):
        #grad_output = grad_output[0].mean(dim=0, keepdim=False)
        gradient_tape.append(grad_output[0].float().cpu())
        gradient_tape_abs.append(grad_output[0].abs().float().cpu())
    
    def register_hooks(model):
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                # Guardamos el handle para poder eliminarlo luego
                handle = layer.register_full_backward_hook(get_gradients)
                handles.append(handle)
            register_hooks(layer)
    
    register_hooks(model)
    
    # Forward pass
    for i, (images, masks) in enumerate(data_loader):
        images = images.to(CUDA)
        masks = masks.to(CUDA)
        with autocast(device_type="cuda", dtype=torch.float16):
            output = model(images)
    
    # Backward pass
    loss = eval_model(
                    scores=output,
                    target=masks,
                    metrics=["iou"],
                    clone=False
                )[0]
    loss.backward()
    
    # Eliminar los hooks
    for handle in handles:
        handle.remove()
        
    return gradient_tape, gradient_tape_abs

def score_model(model, data_loader):
    # Obtener el tape de gradientes
    tape, idk = get_tape(model, data_loader.train)
    score = 0
    for item1, item2 in tqdm(zip(tape, idk), total=len(tape)):
        # Obtener la alineación de los gradientes
        alignment = cosine_similarity_extended(item1)
        idk_metric = cosine_similarity_extended(item2)
        # Score
        print(f"Alignment score for the layer: {alignment}")
        print(f"the other parameter: {idk_metric}")
    
    return score

BATCH_SIZE = 5
# Data loader para los experimentos
data_loader_args, kwargs = TorchDataLoader.get_args({"dataset_len": 20, "batch_size": BATCH_SIZE})
data_loader = TorchDataLoader("carvana", **data_loader_args)

prediction = []
ground_truth = []
df = df.sort_values(by="val_iou", ascending=True)

for i, item in df.iterrows():
    c = Chromosome()
    model = c.get_unet(seed=int(item["seed"]))
    
    try:
        score = score_model(model, data_loader)
        prediction.append(score)
        ground_truth.append(item["val_iou"])
    except Exception as e:
        print(f"Error en el modelo {item['seed']}: {e}")
        continue

exit()
# Normalizar los scores
prediction = np.array(prediction)
prediction = 1- ((prediction - prediction.min()) / (prediction.max() - prediction.min()))
ground_truth = np.array(ground_truth)
ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())

# Graficar los resultados
plt.scatter(prediction, ground_truth)
plt.xlabel("Predicción")
plt.ylabel("Ground Truth")
plt.title("Predicción vs Ground Truth")
plt.savefig("pred_vs_gt.png")
plt.clf()

plt.plot(prediction, label="Predicción")
plt.plot(ground_truth, label="Ground Truth")
plt.xlabel("Modelo")
plt.ylabel("Score")
plt.title("Predicción vs Ground Truth")
plt.legend()
plt.savefig("pred_vs_gt_line.png")