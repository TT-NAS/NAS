"""
Módulo con clases para cargar los distintos datasets de segmentación de imágenes disponibles

Clases
------
- TorchDataset: Superclase para datasets en PyTorch que implementa los métodos __len__ y __getitem__
- COCODataset: Clase para cargar un dataset de COCO
- CarvanaDataset: Clase para cargar el dataset de Carvana
- BasicDataset: Clase para cargar un dataset sencillo con los directorios `"images"` y `"masks"`
"""
import os
from typing import Union

import torch
from torch.utils.data import Dataset

from ..constants import WIDTH, HEIGHT


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
        from ..functions import get_data

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
        from ..functions import get_data

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
        from ..functions import get_data

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
