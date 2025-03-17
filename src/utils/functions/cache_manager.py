"""
Módulo con funciones para el manejo de la caché de tensores de imágenes y máscaras

Funciones
---------
- get_images: Obtiene los nombres de archivo de todas las imágenes del dataset
- get_images_and_masks: Carga los nombres de archivo de las imágenes y máscaras del dataset
- get_cache: Obtiene la lista de tensores de imágenes y máscaras de la caché si es válida
- get_data: Obtiene los datos del dataset en forma de una lista con los nombres de archivo de los tensores
"""
import os
from typing import Optional

import torch
from torchvision import transforms as T
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from ..constants import COCO_IDS


def get_images(images_path: str, train: bool, dataset_default_len: int) -> list[str]:
    """
    Obtiene los nombres de archivo de todas las imágenes del dataset

    Parameters
    ----------
    images_path : str
        Ruta de las imágenes
    train : bool
        Si se cargan las imágenes de entrenamiento o de prueba
    dataset_default_len : int
        Número de imágenes en el dataset

    Returns
    -------
    list
        Lista de nombres de archivo de las imágenes

    Raises
    ------
    ValueError
        Si el número de imágenes del dataset y de imágenes en el directorio no coincide
    """
    images = sorted(os.listdir(images_path))

    if not train:
        return images

    if len(images) != dataset_default_len:
        raise ValueError(
            "El número de imágenes del dataset y de imágenes en el directorio no coincide"
        )

    return images


def get_images_and_masks(images_path: str, masks_path: str, train: bool,
                         dataset_default_len: int, identifier: str, suffix_img: str,
                         suffix_mask: Optional[str] = None) -> tuple[list[str], Optional[list[str]]]:
    """
    Carga los nombres de archivo de las imágenes y máscaras del dataset

    Parameters
    ----------
    images_path : str
        Ruta de las imágenes
    masks_path : str
        Ruta de las máscaras
    train : bool
        Si se cargan las imágenes de entrenamiento o de prueba
    dataset_default_len : int
        Número de imágenes en el dataset
    identifier : str
        Identificador del dataset
    suffix_img : str
        Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
    suffix_mask : Optional[str], optional
        Sufijo a colocar para obtener el nombre de una máscara a partir de un elemento, by default `None`

    Returns
    -------
    tuple
        (Lista de imágenes, lista de máscaras), la lista de máscaras puede ser None

    Raises
    ------
    ValueError
        Si el número de imágenes y máscaras no coincide o si los nombres no coinciden
    """
    images = get_images(
        images_path=images_path,
        train=train,
        dataset_default_len=dataset_default_len
    )

    if not train or identifier in COCO_IDS.values():
        return images, None

    masks = sorted(os.listdir(masks_path))

    if len(images) != len(masks):
        raise ValueError("El número de imágenes y máscaras no coincide")

    for i, image in enumerate(images):
        if suffix_img:
            image_name = image.replace(suffix_img, suffix_mask)
        else:
            image_name = image + suffix_mask

        if masks[i] != image_name:
            raise ValueError(
                f"La imágen {image} con nombre de máscara {image_name} "
                f"no coincide con la máscara {masks[i]}"
            )

    return images, masks


def get_cache(images: list[str], cache_path: str,
              suffix_img: str, suffix_tensor: str) -> Optional[list[str]]:
    """
    Obtiene la lista de tensores de imágenes y máscaras de la caché si es válida

    Parameters
    ----------
    images : list
        Lista de nombres de archivo de las imágenes
    cache_path : str
        Ruta de la caché
    suffix_img : str
        Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
    suffix_tensor : str
        Sufijo a colocar para obtener el nombre de un tensor a partir de un elemento

    Returns
    -------
    Optional[list]
        Lista de tensores si la caché es válida, None en caso contrario
    """
    if not os.path.exists(cache_path):
        return None

    tensors = sorted(os.listdir(cache_path))

    if len(images) != len(tensors):
        return None

    for i, image in enumerate(images):
        if suffix_img:
            image_name = image.replace(suffix_img, suffix_tensor)
        else:
            image_name = image + suffix_tensor

        if tensors[i] != image_name:
            return None

    return tensors


def get_data(train: bool, dataset_default_len, identifier: str, width: int, height: int,
             cache_path: str, suffix_tensor: str, images_path: str, suffix_img: str,
             masks_path: Optional[str] = None, suffix_mask: Optional[str] = None,
             annotations_file: Optional[str] = None) -> list[str]:
    """
    Obtiene los datos del dataset en forma de una lista con los nombres de archivo de los tensores

    Parameters
    ----------
    train : bool
        Si se cargan los datos de entrenamiento o de prueba
    dataset_default_len : int
        Número de imágenes en el dataset
    identifier : str
        Identificador del dataset
    width : int
        Ancho a redimensionar las imágenes
    height : int
        Alto a redimensionar las imágenes
    cache_path : str
        Ruta de la caché
    suffix_tensor : str
        Sufijo a colocar para obtener el nombre de un tensor a partir de un elemento
    images_path : str
        Ruta de las imágenes
    suffix_img : str
        Sufijo a remover para obtener el nombre de un elemento a partir de una imagen
    masks_path : Optional[str], optional
        Ruta de las máscaras, by default `None`
    suffix_mask : Optional[str], optional
        Sufijo a colocar para obtener el nombre de una máscara a partir de un elemento, by default `None`
    annotations_file : Optional[str], optional
        Archivo de anotaciones COCO, by default `None`

    Returns
    -------
    list
        Lista con los nombres de archivo de los tensores

    Raises
    ------
    ValueError
        Si el número de imágenes del dataset y de imágenes en el directorio no coincide
    """
    images, masks = get_images_and_masks(
        images_path=images_path,
        masks_path=masks_path,
        train=train,
        dataset_default_len=dataset_default_len,
        identifier=identifier,
        suffix_img=suffix_img,
        suffix_mask=suffix_mask
    )
    tensors = get_cache(
        images=images,
        cache_path=cache_path,
        suffix_img=suffix_img,
        suffix_tensor=suffix_tensor
    )

    if tensors:
        return tensors

    if os.path.exists(cache_path):
        for file in os.listdir(cache_path):
            os.remove(os.path.join(cache_path, file))

        os.rmdir(cache_path)

    os.makedirs(cache_path)

    if identifier in COCO_IDS.values():
        coco = COCO(annotations_file)
        cat_id = 1 if identifier == "cpp" else 3
        image_ids = sorted(
            coco.getImgIds(catIds=[cat_id]),
            key=lambda img_id: coco.loadImgs(img_id)[0]["file_name"]
        )

        if len(images) != len(image_ids):
            raise ValueError(
                "El número de imágenes del dataset y de imágenes en el directorio no coincide"
            )

        images = image_ids

    transform = T.Compose([
        T.Resize([width, height]),
        T.ToTensor()
    ])
    total = len(images)
    train_str = "Train" if train else "Test"
    progreso = 0

    for i, image in enumerate(images):
        progreso_actual = i / total

        if progreso_actual - progreso >= 0.001 or i == total - 1:
            print(
                f"\rPreprocesando imágenes ({train_str}), "
                "este proceso solo se realiza una vez: "
                f"{progreso_actual * 100:.2f}%", end=""
            )
            progreso = progreso_actual

        if identifier in COCO_IDS.values():
            image_object = coco.loadImgs(image)[0]
            image_name = image_object["file_name"]
        else:
            image_name = image

        image_path = os.path.join(images_path, image_name)
        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = transform(image_tensor)

        if train:
            if identifier in COCO_IDS.values():
                ann_ids = coco.getAnnIds(
                    imgIds=image_object["id"],
                    catIds=[cat_id]
                )
                annotations = coco.loadAnns(ann_ids)
                mask = np.zeros(
                    (image_object["height"], image_object["width"]),
                    dtype=np.float32
                )

                for ann in annotations:
                    mask = np.maximum(mask, coco.annToMask(ann))

                mask = Image.fromarray(mask).resize(
                    (width, height), Image.NEAREST)
                mask = torch.tensor(np.array(mask), dtype=torch.float32)
                mask = mask.unsqueeze(0)
            else:
                mask_path = os.path.join(masks_path, masks[i])
                mask = Image.open(mask_path).convert("L")
                mask = transform(mask)
                mask /= mask.max()

            tensor = torch.cat((image_tensor, mask))
        else:
            tensor = image_tensor

        if suffix_img:
            tensor_name = image_name.replace(suffix_img, suffix_tensor)
        else:
            tensor_name = image_name + suffix_tensor

        torch.save(tensor, os.path.join(
            cache_path,
            tensor_name
        ))

    print(
        f"\rPreprocesando imágenes ({train_str}), "
        "este proceso solo se realiza una vez: "
        "Completado!"
    )

    return sorted(os.listdir(cache_path))
