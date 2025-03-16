import os
from typing import Union, Optional

import torch
from torch import nn, Tensor
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

from .constants import (
    COCO_PEOPLE_DATASET_LENGTH, COCO_CAR_DATASET_LENGTH, CARVANA_DATASET_LENGTH,
    ROAD_DATASET_LENGTH, CAR_DATASET_LENGTH,

    COCO_BATCH_SIZE, CARVANA_BATCH_SIZE,
    ROAD_BATCH_SIZE, CAR_BATCH_SIZE, SHOW_SIZE,

    COCO_PEOPLE_DATA_PATH, COCO_CAR_DATA_PATH, CARVANA_DATA_PATH,
    ROAD_DATA_PATH, CAR_DATA_PATH,

    WIDTH, HEIGHT, CHANNELS,

    COCO_IDS
)


class TorchDataset(Dataset):
    """
    Superclase para datasets en PyTorch que implementa los métodos __len__ y __getitem__
    """

    def __init__(self):
        """
        Inicializa la estructura base del dataset
        """
        self.train = None
        self.cache_path = None
        self.tensors = None

    def __len__(self) -> int:
        """
        Devuelve el número de elementos en el dataset

        Returns
        -------
        int
            Número de elementos en el dataset.
        """
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Devuelve una imagen y su máscara si el dataset es de entrenamiento

        Parameters
        ----------
        idx : int
            Índice de la imagen a cargar

        Returns
        -------
        tuple or Tensor
            (Imagen, Máscara) si el dataset es de entrenamiento, Imagen si es de prueba
        """
        tensor = torch.load(os.path.join(self.cache_path, self.tensors[idx]))
        image = tensor[:3]

        if self.train:
            mask = tensor[3:]
            return image, mask

        return image


class COCODataset(TorchDataset):
    def __init__(self, train: bool, data_path: str, dataset_len: int,
                 imgs_width: int = WIDTH, imgs_height: int = HEIGHT, **kwargs: str):
        """
        Clase para cargar un dataset de COCO

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        data_path : str
            Ruta de los datos
        dataset_len : int
            Número de imágenes a cargar
        imgs_width : int, optional
            Ancho a redimensionar las imágenes, by default `WIDTH`
        imgs_height : int, optional
            Alto a redimensionar las imágenes, by default `HEIGHT`
        **kwargs : str
            Argumentos adicionales para el la carga de los datos:
            - dataset_default_len : (int) Número de imágenes en el dataset
            - identifier : (str) Identificador del dataset
            - suffix_tensor : (str) Sufijo a colocar para obtener el nombre de un tensor a partir de un elemento
            - suffix_img : (str) Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
        """
        from .torch_functions import get_data

        super().__init__()

        self.train = train

        if train:
            self.cache_path = os.path.join(
                data_path,
                f"__cache_{imgs_width}x{imgs_height}__",
                "train_val"
            )
            images_path = os.path.join(data_path, "train2017")
            annotations_file = os.path.join(
                data_path,
                "annotations",
                "instances_train2017.json"
            )
        else:
            self.cache_path = os.path.join(
                data_path,
                f"__cache_{imgs_width}x{imgs_height}__",
                "test"
            )
            images_path = os.path.join(data_path, "test2017")
            annotations_file = os.path.join(
                data_path,
                "annotations",
                "instances_test2017.json"
            )

        kwargs.pop("suffix_mask", None)
        self.tensors = get_data(
            train=train,
            width=imgs_width,
            height=imgs_height,
            cache_path=self.cache_path,
            images_path=images_path,
            annotations_file=annotations_file,
            **kwargs
        )

        if train:
            self.tensors = self.tensors[:dataset_len]


class CarvanaDataset(TorchDataset):
    """
    Clase para cargar el dataset de Carvana.
    """

    def __init__(self, train: bool, data_path: str, dataset_len: int, imgs_width: int = WIDTH,
                 imgs_height: int = HEIGHT, **kwargs: Union[int, str]):
        """
        Clase para cargar el dataset de Carvana

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        data_path : str
            Ruta de los datos
        dataset_len : int
            Número de imágenes a cargar
        imgs_width : int, optional
            Ancho a redimensionar las imágenes, by default `WIDTH`
        imgs_height : int, optional
            Alto a redimensionar las imágenes, by default `HEIGHT`
        **kwargs : int or str
            Argumentos adicionales para el la carga de los datos:
            - dataset_default_len : (int) Número de imágenes en el dataset
            - identifier : (str) Identificador del dataset
            - suffix_tensor : (str) Sufijo a colocar para obtener el nombre de un tensor a partir de un elemento
            - suffix_img : (str) Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
            - suffix_mask : (str) Sufijo a colocar para obtener el nombre de una máscara a partir de un elemento
        """
        from .torch_functions import get_data

        super().__init__()

        self.train = train

        if train:
            self.cache_path = os.path.join(
                data_path,
                f"__cache_{imgs_width}x{imgs_height}__",
                "train_val"
            )
            images_path = os.path.join(data_path, "train")
            masks_path = os.path.join(data_path, "train_masks")
        else:
            self.cache_path = os.path.join(
                data_path,
                f"__cache_{imgs_width}x{imgs_height}__",
                "test"
            )
            images_path = os.path.join(data_path, "test")
            masks_path = None

        self.tensors = get_data(
            train=train,
            width=imgs_width,
            height=imgs_height,
            cache_path=self.cache_path,
            images_path=images_path,
            masks_path=masks_path,
            **kwargs
        )

        if train:
            self.tensors = self.tensors[:dataset_len]


class BasicDataset(TorchDataset):
    """
    Clase para cargar un dataset sencillo con el formato de directorios:
    - data_path
        - images
        - masks
    """

    def __init__(self, train: bool, data_path: str, dataset_len: int, test_prop: float = 0.2,
                 imgs_width: int = WIDTH, imgs_height: int = HEIGHT, **kwargs: Union[int, str]):
        """
        Clase para cargar un dataset sencillo con el formato de directorios:
        - data_path
            - images
            - masks

        Parameters
        ----------
        train : bool
            Si se cargan los datos de entrenamiento o de prueba
        data_path : str
            Ruta de los datos
        dataset_len : int
            Número de imágenes a cargar
        test_prop : float, optional
            Proporción de imágenes que se usarán para test, by default `0.2`
        imgs_width : int, optional
            Ancho a redimensionar las imágenes, by default `WIDTH`
        imgs_height : int, optional
            Alto a redimensionar las imágenes, by default `HEIGHT`
        **kwargs : int or str
            Argumentos adicionales para el la carga de los datos:
            - dataset_default_len : (int) Número de imágenes en el dataset
            - identifier : (str) Identificador del dataset
            - suffix_tensor : (str) Sufijo a colocar para obtener el nombre de un tensor a partir de un elemento
            - suffix_img : (str) Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
            - suffix_mask : (str) Sufijo a colocar para obtener el nombre de una máscara a partir de un elemento
        """
        from .torch_functions import get_data

        super().__init__()

        self.train = train
        self.cache_path = os.path.join(
            data_path,
            f"__cache_{imgs_width}x{imgs_height}__",
            "train_val_test"
        )
        images_path = os.path.join(data_path, "images")
        masks_path = os.path.join(data_path, "masks")

        train_and_val_prop = 1 - test_prop
        split_index = int(dataset_len * train_and_val_prop)

        self.tensors = get_data(
            train=train,
            width=imgs_width,
            height=imgs_height,
            cache_path=self.cache_path,
            images_path=images_path,
            masks_path=masks_path,
            **kwargs
        )

        if train:
            self.tensors = self.tensors[:split_index]
        else:
            self.tensors = self.tensors[split_index:dataset_len]


class TorchDataLoader:
    """
    Wrapper de los DataLoaders de train, validation y test para un dataset
    """

    ARGS = [
        "batch_size",
        "train_val_prop",
        "data_path",
        "dataset_len",
        "test_prop",
        "img_width",
        "img_height"
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
                - "coco-people"
                - "coco-car"
                - "carvana"
                - "road"
                - "car"
        batch_size : int, optional
            Tamaño del batch, by default `BATCH_SIZE`
        train_val_prop : float, optional
            Proporción que se usará entre train y validation, by default `0.8`
        **kwargs : T.Compose or str or int
            Argumentos adicionales para el Dataset:
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - img_width : (int) Ancho a redimensionar las imágenes
            - img_height : (int) Alto a redimensionar las imágenes

        Raises
        ------
        ValueError
            Si el nombre del dataset ingresado no es válido
        """
        if dataset_class == "coco-people" or dataset_class == COCO_IDS["people"]:
            dataset_class = COCODataset
            default_len = COCO_PEOPLE_DATASET_LENGTH
            default_path = COCO_PEOPLE_DATA_PATH
            default_batch_size = COCO_BATCH_SIZE
            suffix_img = ".jpg"
            suffix_mask = None
            self.identifier = COCO_IDS["people"]  # cpp
        elif dataset_class == "coco-car" or dataset_class == COCO_IDS["car"]:
            dataset_class = COCODataset
            default_len = COCO_CAR_DATASET_LENGTH
            default_path = COCO_CAR_DATA_PATH
            default_batch_size = COCO_BATCH_SIZE
            suffix_img = ".jpg"
            suffix_mask = None
            self.identifier = COCO_IDS["car"]  # cca
        elif dataset_class == "carvana" or dataset_class == "cvn":
            dataset_class = CarvanaDataset
            default_len = CARVANA_DATASET_LENGTH
            default_path = CARVANA_DATA_PATH
            default_batch_size = CARVANA_BATCH_SIZE
            suffix_img = ".jpg"
            suffix_mask = "_mask.gif"
            self.identifier = "cvn"
        elif dataset_class == "road" or dataset_class == "rd":
            dataset_class = BasicDataset
            default_len = ROAD_DATASET_LENGTH
            default_path = ROAD_DATA_PATH
            default_batch_size = ROAD_BATCH_SIZE
            suffix_img = ""
            suffix_mask = ".png"
            self.identifier = "rd"
        elif dataset_class == "car" or dataset_class == "car":
            dataset_class = BasicDataset
            default_len = CAR_DATASET_LENGTH
            default_path = CAR_DATA_PATH
            default_batch_size = CAR_BATCH_SIZE
            suffix_img = ".jpg"
            suffix_mask = ".png"
            self.identifier = "car"
        else:
            raise ValueError("Invalid dataset class")

        dataset_path = kwargs.pop("data_path", default_path)
        dataset_len = kwargs.pop("dataset_len", None)

        if dataset_len:
            dataset_len = min(dataset_len, default_len)
        else:
            dataset_len = default_len

        dataset = dataset_class(
            train=True,
            data_path=dataset_path,
            dataset_len=dataset_len,
            dataset_default_len=default_len,
            suffix_img=suffix_img,
            suffix_mask=suffix_mask,
            suffix_tensor=".pt",
            identifier=self.identifier,
            **kwargs
        )
        test_dataset = dataset_class(
            train=False,
            data_path=dataset_path,
            dataset_len=dataset_len,
            dataset_default_len=default_len,
            suffix_img=suffix_img,
            suffix_mask=suffix_mask,
            suffix_tensor=".pt",
            identifier=self.identifier,
            **kwargs
        )

        TRAIN_SIZE = int(train_val_prop * len(dataset))
        VAL_SIZE = len(dataset) - TRAIN_SIZE

        train_dataset, val_dataset = random_split(
            dataset=dataset,
            lengths=[TRAIN_SIZE, VAL_SIZE]
        )

        if batch_size is None:
            batch_size = default_batch_size

        self.train = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        self.validation = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        self.test = DataLoader(
            dataset=test_dataset,
            batch_size=SHOW_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    def __str__(self) -> str:
        return self.identifier

    @staticmethod
    def get_args(kwargs: dict[str, any]) -> tuple[dict[str,
                                                       Union[str, int, float, T.Compose, None]],
                                                  dict[str, any]]:
        """
        Obtiene los argumentos para el DataLoader y para el Dataset y los separa
        de otros argumentos adicionales

        Parameters
        ----------
        kwargs : any
            Todos los argumentos ingresados

        Returns
        -------
        tuple
            (Argumentos para el DataLoader, Argumentos adicionales)
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
        "linear": nn.Identity,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softsign": nn.Softsign,
        "softmax": nn.Softmax
    }

    def __init__(
        self,
        decoded_chromosome:
            tuple[list[Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
                                      str],
                                tuple[list[Optional[tuple[int, int, str]]],
                                      bool]]]],
                  list[Optional[tuple[int, int, str]]]],
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
            Número de canales de entrada, by default `CHANNELS`
        """
        super().__init__()

        layers, bottleneck = decoded_chromosome
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        concats_sizes: list[int] = []

        for layer in layers:
            if layer is None:
                continue

            (convs, pooling), _ = layer
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

        for layer in layers[::-1]:
            if layer is None:
                continue

            _, (convs, concat) = layer
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

        # Ultima capa convolucional
        self.decoder.append(
            nn.Conv2d(
                in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1
            )
        )

    def build_convs(self, convs: list[Optional[tuple[int, int, str]]],
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

        for i, conv in enumerate(convs):
            if conv is None:
                continue

            filters, kernel_size, activation = conv
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
