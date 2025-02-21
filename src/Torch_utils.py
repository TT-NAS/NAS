"""
Módulo con funciones y clases útiles para trabajar con PyTorch:

Clases
------
- CarvanaDataset: Clase para cargar el dataset de Carvana
- RoadDataset: Clase para cargar el dataset de carreteras
- TorchDataLoader: Wrapper de los DataLoaders de train, validation y test para un dataset
    - No es necesario instanciar esta clase, se puede usar directamente un DataLoader de PyTorch,
      pero esta clase facilita el proceso de cargar los datos
- UNet: Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado

Funciones
---------
- plot_batch: Muestra o guarda un conjunto de imágenes y máscaras junto con sus dimensiones
- plot_results: Muestra o guarda los resultados de un modelo en un conjunto de imágenes
- dice_loss: Calcula la pérdida de Dice
- dice_crossentropy_loss: Calcula la pérdida de Dice-Crossentropy
- iou_loss: Calcula la pérdida de IoU
- accuracy: Calcula la precisión de la máscara predicha
- eval_model: Evalúa un modelo en base a una lista de métricas
- train_model: Entrena un modelo UNet
- save_model: Guarda un modelo en un archivo
- synflow: Versión de copilot
- gradient_scorer_pytorch: No jala
"""
import os
from typing import Union, Optional

# Torch
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
# Images
from PIL import Image
import matplotlib.pyplot as plt


# ========================
# NOTE: Constantes
# ========================
torch.manual_seed(42)
CARVANA_BATCH_SIZE = 32
ROAD_BATCH_SIZE = 1
SHOW_SIZE = 32
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
CARVANA_DATA_PATH = "./carvana-dataset/"
ROAD_DATA_PATH = "./road-dataset/"
MODELS_PATH = "./models/"
IMGS_PATH = "./imgs/"
TRANSFORM = T.Compose([
    T.Resize([WIDTH, HEIGHT]),
    T.ToTensor()
])
CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU")


# ========================
# NOTE: Clases
# ========================
class CarvanaDataset(Dataset):
    """
    Clase para cargar el dataset de Carvana
    """

    def __init__(self, train: bool, transform: T.Compose = TRANSFORM, **_):
        """
        Clase para cargar el dataset de Carvana

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        transform : T.Compose, optional
            Transformaciones a aplicar a las imágenes, by default TRANSFORM
        """
        self.transform = transform

        CARVANA_TRAIN_PATH = os.path.join(CARVANA_DATA_PATH, "train")
        CARVANA_MASKS_PATH = os.path.join(CARVANA_DATA_PATH, "train_masks")
        CARVANA_TEST_PATH = os.path.join(CARVANA_DATA_PATH, "test")

        if train:
            self.train = True

            self.image_paths = CARVANA_TRAIN_PATH
            self.mask_paths = CARVANA_MASKS_PATH
            self.images = sorted(os.listdir(self.image_paths))
            self.masks = sorted(os.listdir(self.mask_paths))

            assert (
                len(self.images) == len(self.masks)
            ), "El número de imágenes y máscaras no coincide"
        else:
            self.train = False

            self.image_paths = CARVANA_TEST_PATH
            self.images = sorted(os.listdir(self.image_paths))
            self.masks = None

    def __len__(self) -> int:
        """
        Devuelve el número de imágenes en el dataset

        Returns
        -------
        int
            Número de imágenes
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Union[tuple[Tensor, Tensor], Tensor]:
        """
        Devuelve una imagen y su máscara si el dataset es de entrenamiento

        Parameters
        ----------
        idx : int
            Índice de la imagen a cargar

        Returns
        -------
        tuple or Tensor
            (Imagen, Máscara) si el dataset es de entrenamiento, (Imagen) si es de prueba
        """
        image_path = os.path.join(self.image_paths, self.images[idx])
        image = Image.open(image_path)
        image = self.transform(image)

        if self.train:
            mask_path = os.path.join(self.mask_paths, self.masks[idx])
            mask = Image.open(mask_path)
            mask = self.transform(mask)
            mask /= mask.max().item()

            return image, mask
        else:
            return image


class RoadDataset(Dataset):
    """
    Clase para cargar el dataset de carreteras
    """

    def __init__(self, train: bool, test_prop: float, transform: T.Compose = TRANSFORM):
        """
        Clase para cargar el dataset de carreteras

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        test_prop : float
            Proporción de imágenes que se usarán para test
        transform : T.Compose, optional
            Transformaciones a aplicar a las imágenes, by default TRANSFORM
        """
        self.transform = transform

        ROAD_TRAIN_PATH = os.path.join(ROAD_DATA_PATH, "images")
        ROAD_MASKS_PATH = os.path.join(ROAD_DATA_PATH, "masks")

        len_dataset = len(os.listdir(ROAD_TRAIN_PATH))
        train_and_val_prop = 1 - test_prop
        split_index = int(len_dataset * train_and_val_prop)

        if train:
            self.image_paths = ROAD_TRAIN_PATH
            self.mask_paths = ROAD_MASKS_PATH
            self.images = sorted(os.listdir(self.image_paths))[:split_index]
            self.masks = sorted(os.listdir(self.mask_paths))[:split_index]

            assert (
                len(self.images) == len(self.masks)
            ), "El número de imágenes y máscaras no coincide"
        else:
            self.image_paths = ROAD_TRAIN_PATH
            self.images = sorted(os.listdir(self.image_paths))[split_index:]
            self.masks = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Optional[Tensor]]:
        image_path = os.path.join(self.image_paths, self.images[idx])
        image = Image.open(image_path)
        image = self.transform(image)

        if self.masks:
            mask_path = os.path.join(self.mask_paths, self.masks[idx])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform(mask)
            mask /= mask.max().item()

            return image, mask
        else:
            return image


class TorchDataLoader:
    """
    Wrapper de los DataLoaders de train, validation y test para un dataset
    """

    def __init__(self, dataset_class: Union[Dataset, str], batch_size: Optional[int] = None, train_val_prop: float = 0.8, test_prop: float = 0.2, **kwargs: T.Compose):
        """
        Wrapper de los DataLoaders de train, validation y test para un dataset

        Parameters
        ----------
        dataset_class : Dataset or str
            Clase del dataset a cargar o nombre del dataset

            Opciones:
                - "carvana"
                - "road"
        batch_size : int, optional
            Tamaño del batch, by default BATCH_SIZE
        train_val_prop : float, optional
            Proporción que se usará entre train y validation, by default 0.8
        test_prop : float, optional
            Proporción que se usará entre el conjunto de entrenamiento (train y validation) y test, by default 0.2
        **kwargs : T.Compose or float
            Argumentos adicionales para el dataset:
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes

        Raises
        ------
        ValueError
            Si el nombre del dataset ingresado no es válido
        """
        if isinstance(dataset_class, str):
            if dataset_class == "carvana" or dataset_class == "c":
                dataset_class = CarvanaDataset
                self.identifier = "c"
            elif dataset_class == "road" or dataset_class == "r":
                dataset_class = RoadDataset
                self.identifier = "r"
            else:
                raise ValueError("Invalid dataset class")
        else:
            if isinstance(dataset_class, CarvanaDataset):
                self.identifier = "c"
            elif isinstance(dataset_class, RoadDataset):
                self.identifier = "r"
            else:
                self.identifier = "0"

        dataset = dataset_class(
            train=True,
            test_prop=test_prop,
            **kwargs
        )
        test_dataset = dataset_class(
            train=False,
            test_prop=test_prop,
            **kwargs
        )

        TRAIN_SIZE = int(train_val_prop * len(dataset))
        VAL_SIZE = len(dataset) - TRAIN_SIZE

        train_dataset, val_dataset = random_split(
            dataset=dataset,
            lengths=[TRAIN_SIZE, VAL_SIZE]
        )

        if batch_size is None:
            if isinstance(dataset, CarvanaDataset):
                batch_size = CARVANA_BATCH_SIZE
            elif isinstance(dataset, RoadDataset):
                batch_size = ROAD_BATCH_SIZE
            else:
                batch_size = 1

        self.train = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.validation = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.test = DataLoader(
            dataset=test_dataset,
            batch_size=SHOW_SIZE,
            shuffle=False
        )

    def __str__(self) -> str:
        return self.identifier


class UNet(nn.Module):
    """
    Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "selu": nn.SELU,
        "elu": nn.ELU,
        # No existe `nn.Exponential`, pero `Sigmoid` es una opción cercana
        "exponential": nn.Sigmoid,
        "linear": nn.Identity,
    }

    def __init__(
        self,
        decoded_chromosome:
            tuple[list[tuple[tuple[list[tuple[int, int, str]],
                                   str],
                             tuple[list[tuple[int, int, str]],
                                   bool]]],
                  list[tuple[int, int, str]]],
        in_channels: int = CHANNELS
    ):
        """
        Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado

        Parameters
        ----------
        decoded_chromosome : tuple
            Cromosoma decodificado con la arquitectura deseada
        in_channels : int, optional
            Número de canales de entrada, by default CHANNELS
        """
        super().__init__()
        layers, bottleneck = decoded_chromosome
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        concats_sizes: list[int] = []

        for encoder, _ in layers:
            convs, pooling = encoder
            convs_layer, in_channels = self.build_convs(convs, in_channels)
            self.encoder.append(convs_layer)
            concats_sizes.append(in_channels)

            if pooling == "max":
                self.encoder.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            else:
                self.encoder.append(
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )

        # Bottleneck
        self.bottleneck, in_channels = self.build_convs(
            bottleneck,
            in_channels
        )

        # Decoder
        self.concat_or_not = []

        for _, decoder in layers[::-1]:
            convs, concat = decoder
            concat_size = concats_sizes.pop()

            if concat:
                out_channels = max(1, in_channels // 2)
            else:
                out_channels = in_channels

            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2
                )
            )

            if concat:
                in_channels = out_channels + concat_size
            else:
                in_channels = out_channels

            convs_layer, in_channels = self.build_convs(convs, in_channels)
            self.decoder.append(convs_layer)
            self.concat_or_not.append(concat)

        # Add the last convolutional layer
        self.decoder.append(
            nn.Conv2d(
                in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1
            )
        )

    def build_convs(self, convs: list[tuple[int, int, str]], in_channels: int) -> tuple[nn.Sequential, int]:
        """
        Construye una secuencia de convoluciones con batch normalization

        Parameters
        ----------
        convs : list
            Lista de convoluciones
        in_channels : int
            Número de canales de entrada

        Returns
        -------
        tuple
            (Secuencia de convoluciones, Número de canales de entrada actualizado)
        """
        convs_layer = nn.Sequential()

        for i, (filters, kernel_size, activation) in enumerate(convs):
            padding = kernel_size // 2
            convs_layer.add_module(
                f"conv_{filters}_{i}",
                nn.Conv2d(in_channels, filters, kernel_size, padding=padding)
            )
            convs_layer.add_module(
                f"batch_norm_{i}",
                nn.BatchNorm2d(filters)
            )

            if activation in UNet.ACTIVATIONS:
                if activation == "softmax":
                    activ = UNet.ACTIVATIONS[activation](dim=1)
                else:
                    activ = UNet.ACTIVATIONS[activation]()
            else:
                activ = nn.Identity()

            convs_layer.add_module(
                f"act_{activation}_{i}",
                activ
            )
            in_channels = filters

        return convs_layer, in_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Propagación hacia adelante de la red

        Parameters
        ----------
        x : Tensor
            Imagen de entrada

        Returns
        -------
        Tensor
            Imagen segmentada
        """
        # Encoder
        # print(x.shape)
        concat_data = []
        num_concat = 0

        for layer in self.encoder:
            x = layer(x)
            # print(x.shape)

            # Si es una capa de convoluciones (no pooling)
            if isinstance(layer, nn.Sequential):
                if self.concat_or_not[::-1][num_concat]:
                    concat_data.append(x)
                else:
                    concat_data.append(None)

                num_concat += 1

        # Bottleneck
        x = self.bottleneck(x)
        # print(x.shape)

        # Decoder
        num_concat = 0

        for layer in self.decoder:
            x = layer(x)
            # print(x.shape)

            # Si es una up-convolution
            if isinstance(layer, nn.ConvTranspose2d):
                concat_layer = concat_data.pop()

                if self.concat_or_not[num_concat]:
                    # print(concat_layer.shape)
                    x = torch.cat([concat_layer, x], dim=1)
                    # print(x.shape)

                num_concat += 1

        return x


class Synflow:
    def __init__(self, model: UNet):
        """
        Clase para calcular el puntaje de un modelo utilizando Synflow

        Parameters
        ----------
        model : UNet
            Modelo a evaluar
        """
        self.masked_parameters = [
            (name, param)
            for name, param in model.named_parameters()
        ]

    def score(self, model: UNet, shape: list[int], device: torch.device = CUDA) -> float:
        """
        Calcula el puntaje de un modelo utilizando Synflow

        Parameters
        ----------
        model : UNet
            Modelo a evaluar
        shape : list
            Dimensiones de la imagen de entrada
        device : torch.device, optional
            Dispositivo para realizar los cálculos, by default CUDA

        Returns
        -------
        float
            Puntaje del modelo
        """
        scores = {}

        @torch.no_grad()
        def linearize(model: UNet) -> dict[str, Tensor]:
            signs = {}

            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param).to(device)
                param.abs_()

            return signs

        @torch.no_grad()
        def nonlinearize(model: UNet, signs: dict[str, Tensor]):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        input_dim = shape
        input_tensor = torch.ones([1] + input_dim).to(device)
        model = model.to(device)
        output = model(input_tensor)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

        return sum(torch.sum(score_tensor) for score_tensor in scores.values()).item()


# ========================
# NOTE: Funciones
# ========================
def plot_batch(imgs: Tensor, masks: Tensor, save: bool = False, show_size: int = SHOW_SIZE, name: str = "batch.png", path: str = IMGS_PATH):
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
    print(f" - Imágenes: {imgs.shape}")
    print(f" - Máscaras: {masks.shape}")
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


def synflow(model):
    """
    No c, lo hizo el copilot
    """
    # genera un input aleatorio con la forma correcta
    random_input = torch.randn((1, CHANNELS, HEIGHT, WIDTH))

    # Realiza una pasada hacia adelante para obtener la salida de la red
    output = model(random_input)

    # Calcula la suma de los outputs para obtener un escalar
    loss = output.sum()

    # Calcula los gradientes de la "loss" con respecto a los pesos del modelo
    loss.backward()
    gradients = [param.grad for param in model.parameters()]

    scores = [torch.abs(weight * grad)
              for weight, grad in zip(model.parameters(), gradients)]
    scores = [score.sum() for score in scores]

    return torch.log(sum(scores)).item()


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


if __name__ == "__main__":
    # === Prueba de DataLoader ===
    # Elige el dataset a cargar
    name = "carvana"
    bs = CARVANA_BATCH_SIZE
    # name = "road"
    # bs = ROAD_BATCH_SIZE

    data_loader = TorchDataLoader(name)
    imgs, masks = next(iter(data_loader.train))

    # SHOW_SIZE solo es para mostrar las imágenes de prueba, si se quiere ver un batch de entrenamiento
    # se debe asignar show_size al correspondiente del dataset, por ejemplo, CARVANA_BATCH_SIZE o ROAD_BATCH_SIZE
    plot_batch(imgs, masks, show_size=bs)

    # Comprobación de las dimensiones de los DataLoaders
    train = data_loader.train
    data_train = next(iter(train))
    val = data_loader.validation
    data_val = next(iter(val))
    test = data_loader.test
    data_test = next(iter(test))

    print(
        f"Train: {len(train)},\t length: {len(data_train)},\t shape: {data_train[0].shape}"
    )
    print(
        f"Val:   {len(val)},\t length: {len(data_val)},\t shape: {data_val[0].shape}"
    )
    print(
        f"Test:  {len(test)},\t length: 1,\t shape: {data_test.shape}"
    )
