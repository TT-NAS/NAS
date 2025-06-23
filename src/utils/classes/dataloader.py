"""
Módulo con la clase TorchDataLoader para el manejo de los DataLoaders `train`,
`validation` y `test` de un dataset dado

Clases
------
- TorchDataLoader: Wrapper de los DataLoaders de train, validation y test para un TorchDataset
"""
from typing import Union, Optional

from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset, random_split 

from ..constants import (
    COCO_IDS,

    COCO_BATCH_SIZE, CARVANA_BATCH_SIZE,
    ROAD_BATCH_SIZE, CAR_BATCH_SIZE, SHOW_SIZE,

    ROAD_DATA_PATH, CAR_DATA_PATH,
    COCO_PEOPLE_DATA_PATH, COCO_CAR_DATA_PATH, CARVANA_DATA_PATH,

    ROAD_DATASET_LENGTH, CAR_DATASET_LENGTH,
    COCO_PEOPLE_DATASET_LENGTH, COCO_CAR_DATASET_LENGTH, CARVANA_DATASET_LENGTH,
)
from .datasets import COCODataset, CarvanaDataset, BasicDataset


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
        "img_height",
        "k_folds_subsets"
    ]

    def __init__(self, dataset_class: str, batch_size: Optional[int] = None,
                 train_val_prop: float = 0.8, k_folds_subsets: tuple[Subset] = None,
                 **kwargs: Union[T.Compose, str, int]):
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
        self.full_dataset = dataset

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
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        self.validation = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        self.test = DataLoader(
            dataset=test_dataset,
            batch_size=SHOW_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        if k_folds_subsets:
            self.train = DataLoader(
                dataset=k_folds_subsets[0],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            self.validation = DataLoader(
                dataset=k_folds_subsets[1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
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
