import os
from typing import Union, Optional

import torch
from torch import nn, Tensor
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

from .constants import (
    CARVANA_DATASET_LENGTH, ROAD_DATASET_LENGTH, CAR_DATASET_LENGTH,
    CARVANA_BATCH_SIZE, ROAD_BATCH_SIZE, CAR_BATCH_SIZE, SHOW_SIZE,
    CARVANA_DATA_PATH, ROAD_DATA_PATH, CAR_DATA_PATH,
    CHANNELS,
    TRANSFORM,
    CUDA
)


class CarvanaDataset(Dataset):
    """
    Clase para cargar el dataset de Carvana
    """

    def __init__(self, train: bool, transform: T.Compose = TRANSFORM,
                 data_path: str = CARVANA_DATA_PATH, dataset_len: int = CARVANA_DATASET_LENGTH):
        """
        Clase para cargar el dataset de Carvana

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        transform : T.Compose, optional
            Transformaciones a aplicar a las imágenes, by default TRANSFORM
        data_path : str, optional
            Ruta de los datos, by default CARVANA_DATA_PATH
        dataset_len : int, optional
            Número de imágenes a cargar, by default CARVANA_DATASET_LENGTH
        """
        self.transform = transform

        CARVANA_TRAIN_PATH = os.path.join(data_path, "train")
        CARVANA_MASKS_PATH = os.path.join(data_path, "train_masks")
        CARVANA_TEST_PATH = os.path.join(data_path, "test")

        dataset_len = min(dataset_len, CARVANA_DATASET_LENGTH)

        if train:
            self.train = True

            self.image_paths = CARVANA_TRAIN_PATH
            self.mask_paths = CARVANA_MASKS_PATH
            self.images = sorted(os.listdir(self.image_paths))[:dataset_len]
            self.masks = sorted(os.listdir(self.mask_paths))[:dataset_len]

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

    def __init__(self, train: bool, test_prop: float = 0.2, transform: T.Compose = TRANSFORM,
                 data_path: str = ROAD_DATA_PATH, dataset_len: int = ROAD_DATASET_LENGTH):
        """
        Clase para cargar el dataset de carreteras

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        test_prop : float
            Proporción de imágenes que se usarán para test, by default 0.2
        transform : T.Compose, optional
            Transformaciones a aplicar a las imágenes, by default TRANSFORM
        data_path : str, optional
            Ruta de los datos, by default ROAD_DATA_PATH
        dataset_len : int, optional
            Número de imágenes a cargar, by default ROAD_DATASET_LENGTH
        """
        self.transform = transform

        ROAD_TRAIN_PATH = os.path.join(data_path, "images")
        ROAD_MASKS_PATH = os.path.join(data_path, "masks")

        dataset_len = min(dataset_len, ROAD_DATASET_LENGTH)
        train_and_val_prop = 1 - test_prop
        split_index = int(dataset_len * train_and_val_prop)

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
            self.images = sorted(
                os.listdir(self.image_paths)
            )[split_index:dataset_len]
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

        if self.masks:
            mask_path = os.path.join(self.mask_paths, self.masks[idx])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform(mask)
            mask /= mask.max().item()

            return image, mask
        else:
            return image


class CarDataset(Dataset):
    """
    Clase para cargar el dataset de carros
    """

    def __init__(self, train: bool, test_prop: float = 0.2, transform: T.Compose = TRANSFORM,
                 data_path: str = CAR_DATA_PATH, dataset_len: int = CAR_DATASET_LENGTH):
        """
        Clase para cargar el dataset de carros

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        test_prop : float
            Proporción de imágenes que se usarán para test, by default 0.2
        transform : T.Compose, optional
            Transformaciones a aplicar a las imágenes, by default TRANSFORM
        data_path : str, optional
            Ruta de los datos, by default CAR_DATA_PATH
        dataset_len : int, optional
            Número de imágenes a cargar, by default CAR_DATASET_LENGTH
        """
        self.transform = transform

        CAR_TRAIN_PATH = os.path.join(data_path, "Images")
        CAR_MASKS_PATH = os.path.join(data_path, "Masks")

        dataset_len = min(dataset_len, CAR_DATASET_LENGTH)
        train_and_val_prop = 1 - test_prop
        split_index = int(dataset_len * train_and_val_prop)

        if train:
            self.image_paths = CAR_TRAIN_PATH
            self.mask_paths = CAR_MASKS_PATH
            self.images = sorted(os.listdir(self.image_paths))[:split_index]
            self.masks = sorted(os.listdir(self.mask_paths))[:split_index]

            assert (
                len(self.images) == len(self.masks)
            ), "El número de imágenes y máscaras no coincide"
        else:
            self.image_paths = CAR_TRAIN_PATH
            self.images = sorted(
                os.listdir(self.image_paths)
            )[split_index:dataset_len]
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

    ARGS = [
        "batch_size",
        "train_val_prop",
        "test_prop",
        "transform",
        "data_path",
        "dataset_len"
    ]

    def __init__(self, dataset_class: str, batch_size: Optional[int] = None,
                 train_val_prop: float = 0.8, **kwargs: Union[T.Compose, str, int]):
        """
        Wrapper de los DataLoaders de train, validation y test para un dataset

        Parameters
        ----------
        dataset_class : str
            Nombre del dataset a cargar

            Opciones:
                - "carvana"
                - "road"
                - "car"
        batch_size : int, optional
            Tamaño del batch, by default BATCH_SIZE
        train_val_prop : float, optional
            Proporción que se usará entre train y validation, by default 0.8
        **kwargs : T.Compose or str or int
            Argumentos adicionales para el dataset:
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar

        Raises
        ------
        ValueError
            Si el nombre del dataset ingresado no es válido
        """
        if dataset_class == "carvana" or dataset_class == "c":
            dataset_class = CarvanaDataset
            self.identifier = "c"
        elif dataset_class == "road" or dataset_class == "r":
            dataset_class = RoadDataset
            self.identifier = "r"
        elif dataset_class == "car" or dataset_class == "ca":
            dataset_class = CarDataset
            self.identifier = "ca"
        else:
            raise ValueError("Invalid dataset class")

        dataset = dataset_class(
            train=True,
            **kwargs
        )
        test_dataset = dataset_class(
            train=False,
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
            elif isinstance(dataset, CarDataset):
                batch_size = CAR_BATCH_SIZE
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

    @staticmethod
    def get_args(kwargs: dict[str, any]) -> tuple[dict[str,
                                                       Union[str, int, float, T.Compose, None]],
                                                  dict[str, any]]:
        """
        Devuelve los argumentos necesarios para instanciar la clase

        Returns
        -------
        tuple
            (Argumentos necesarios para instanciar la clase, argumentos restantes)
        """
        data_loader_args = {}

        if any(arg in kwargs for arg in TorchDataLoader.ARGS):
            data_loader_args = {
                k: v for k, v in kwargs.items()
                if k in TorchDataLoader.ARGS
            }
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in TorchDataLoader.ARGS
            }

        return data_loader_args, kwargs


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
        Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma
        decodificado

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

    def build_convs(self, convs: list[tuple[int, int, str]],
                    in_channels: int) -> tuple[nn.Sequential, int]:
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

    def score(self, model: UNet, shape: list[int]) -> float:
        """
        Calcula el puntaje de un modelo utilizando Synflow

        Parameters
        ----------
        model : UNet
            Modelo a evaluar
        shape : list
            Dimensiones de la imagen de entrada

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
                signs[name] = torch.sign(param).to(CUDA)
                param.abs_()

            return signs

        @torch.no_grad()
        def nonlinearize(model: UNet, signs: dict[str, Tensor]):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        input_dim = shape
        input_tensor = torch.ones([1] + input_dim).to(CUDA)
        model = model.to(CUDA)
        output = model(input_tensor)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

        return sum(torch.sum(score_tensor) for score_tensor in scores.values()).item()
