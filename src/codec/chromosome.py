"""
Módulo que contiene la clase Chromosome, la cual representa un cromosoma de una red UNet,
una misma instancia engloba las 3 codificaciones posibles para el cromosoma:

- Real
- Binaria
- Decodificada

Aunque no todas están definidas desde que se crea el cromosoma, se crean cuando se necesitan.

Para usar la clase basta con crear una instancia de Chromosome, asignarle un cromosoma si se quiere
o una semilla si se quiere generar aleatoriamente, y luego usar los getters o metodos de la clase
para entrenar el modelo, mostrar sus resultados, guardarlo, etc.

# Miembros de la clase Chromosome:

Variables
----------
- seed : Semilla para la generación de números aleatorios
    - Se utiliza cuando el cromosoma se genera aleatoriamente
- max_layers : Número máximo de capas de la red sin contar el bottleneck (`max_layers` no puede ser
               mayor que `codec.MAX_LAYERS`)
- max_conv_per_layer : Número máximo de capas convolucionales por bloque
- data_loader : Identificador del DataLoader con el que se ha evaluado el cromosoma
    - Es un identificador corto que indica el DataLoader utilizado ('cpp' para coco-people,
      'cca' para coco-car, 'cvn' para carvana, 'rd' para road, 'car' para car y '0' para
      no especificado)
    - Se asigna solo al entrenar el modelo, después puede se utiliza para deducir el DataLoader
      a ocupar en métodos como `get_aptitude` o `show_results` si no se proporciona uno
- aptitude : Aptitud del cromosoma
    - Es la pérdida del modelo UNet evaluado con el DataLoader especificado
      o guardado en `data_loader`
- __unet : Modelo UNet creado a partir del cromosoma decodificado
- __decoded : Es el cromosoma en su forma más entendible, con listas y tuplas
- __real : Es el cromosoma en su forma real, con valores entre 0 y 1
- __binary : Es el cromosoma en su forma binaria, con 0s y 1s

# Métodos de la clase Chromosome:

Utils
-----
- __str__ : Representación en string de la clase
- __repr__ : Representación en string de la clase para el debugger
- validate : Valida que el cromosoma cumpla con las condiciones mínimas
- generate_random : Genera un cromosoma decodificado aleatorio

Setters
-------
Funciones para asignar el cromosoma a partir de otro, o para generarlos si no están definidos
(`set_decoded`, `set_real`, `set_binary`, `set_unet`, `set_aptitude`)

Getters
-------
Funciones para obtener el cromosoma en su forma decodificada, real, binaria, el modelo UNet,
la aptitud del cromosoma o el número de capas de la red (`get_decoded`, `get_real`, `get_binary`,
`get_unet`, `get_aptitude`, `get_num_layers`)

Funciones de la UNet
--------------------
Funciones para entrenar, evaluar y guardad el modelo UNet
"""
import time
import random
from typing import Union, Optional

from utils import CUDA, float16
from utils import UNet, TorchDataLoader, autocast
from utils import (
    plot_results,
    train_model, save_model, eval_model,
    remove_checkpoints, set_current_net_binary
)
from .functions import (
    get_num_layers, get_num_convs,
    zip_binary, encode_chromosome,
    unzip_binary, decode_chromosome
)
from .constants import (
    MAX_LAYERS, MAX_CONVS_PER_LAYER,
    LEN_POOLINGS_REAL, LEN_CONVS_REAL, LEN_LAYER_REAL, LEN_CHROMOSOME_REAL,
    LEN_POOLINGS_BIN, LEN_CONVS_BIN, LEN_LAYER_BIN, LEN_CHROMOSOME_BIN,

    VALID_FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS,
    VALID_POOLINGS, CONCATENATION
)


class Chromosome:
    """
    Clase que representa un cromosoma de una red UNet
    """

    def __init__(self, seed: Optional[int] = None,
                 chromosome: Optional[Union[tuple, list, str]] = None,
                 max_layers: int = MAX_LAYERS, max_convs_per_layer: int = MAX_CONVS_PER_LAYER):
        """
        Clase que representa un cromosoma de una red UNet

        Parameters
        ----------
        seed : Optional[int], optional
            Semilla para la generación de números aleatorios, by default `None`
        chromosome : Optional[tuple or list or str], optional
            Cromosoma de un individuo a cargar (decodificado, real o binario), by default `None`
        max_layers : int, optional
            Número máximo de capas de la red sin contar el bottleneck (`max_layers` no puede ser
            mayor que `codec.MAX_LAYERS`), by default `MAX_LAYERS`
        max_convs_per_layer : int
            Número máximo de capas convolucionales por bloque (`max_convs_per_layer` no puede ser
            mayor que `codec.MAX_CONVS_PER_LAYER`), by default `MAX_CONVS_PER_LAYER`
        """
        if max_layers > MAX_LAYERS:
            raise ValueError(
                "El número máximo de capas supera el máximo permitido"
            )

        if max_convs_per_layer > MAX_CONVS_PER_LAYER:
            raise ValueError(
                "El número máximo de convoluciones por capa supera el máximo permitido"
            )

        self.seed = seed
        self.max_layers = max_layers
        self.max_convs_per_layer = max_convs_per_layer

        # __decoded será el cromosoma como tal, __real y __binary son solo auxiliares y caché
        self.data_loader = "0"
        self.data_loader_args = {}
        self.aptitude = None
        self.__decoded = ()
        self.__unet = None
        self.__real = []
        self.__binary = str()

        if chromosome is not None:
            if not chromosome:
                raise ValueError("El cromosoma no puede estar vacío")

            if isinstance(chromosome, tuple):
                self.__decoded = chromosome
            elif isinstance(chromosome, list):
                self.__real = chromosome
            elif isinstance(chromosome, str):
                self.__binary = unzip_binary(chromosome)
            else:
                raise ValueError(
                    "El cromosoma debe ser una tupla, lista o string"
                )

            self.validate()

    # ========================
    # NOTE: Utils de la clase
    # ========================
    def __str__(self) -> str:
        """
        Devuelve una representación en string del cromosoma

        Returns
        -------
        str
            Representación en string del cromosoma
        """
        seed = self.seed if self.seed is not None else 'N'
        binary = self.get_binary(zip=True)
        transform = repr(self.data_loader_args.get(
            "transform", "0"
        )).replace("\n", "").replace(" ", "").replace("'", "")
        data_path = self.data_loader_args.get(
            "data_path", "0"
        ).replace("\n", "")

        return f"s{seed}-{binary}-dl{self.data_loader}-t{transform}-dp{data_path}"

    def __repr__(self) -> str:
        """
        Devuelve la representación de la instancia

        Returns
        -------
        str
            Representación de la instancia
        """
        return (
            "Chromosome("
            f"seed={self.seed}, "
            f"chromosome='{self.get_binary(zip=True)}'"
            ")"
        )

    def validate(self):
        """
        Valida que el cromosoma tenga un tamaño correcto

        Raises
        -------
        ValueError
            Si el cromosoma no cumple con las condiciones necesarias
        """
        num_layers = None

        # Si self.real está definido
        if self.__real:
            if not isinstance(self.__real, list):
                raise ValueError(
                    "El cromosoma real no es una lista"
                )

            if len(self.__real) != LEN_CHROMOSOME_REAL:
                raise ValueError(
                    "El cromosoma real no tiene el tamaño correcto"
                )

            if not all(0 <= v <= 1 for v in self.__real):
                raise ValueError(
                    "Los valores del cromosoma real no están todos en el rango [0, 1]"
                )

            num_layers = get_num_layers(self.__real, "real")

            if num_layers > self.max_layers:
                raise ValueError(
                    "El número de capas supera al máximo del cromosoma"
                )

            for i in range(0, len(self.__real) - LEN_CONVS_REAL, LEN_LAYER_REAL):
                capa = i // LEN_LAYER_REAL
                encoder_convs = self.__real[i:i + LEN_CONVS_REAL]
                decoder_convs = self.__real[
                    i + LEN_CONVS_REAL + LEN_POOLINGS_REAL:
                    i + LEN_CONVS_REAL + LEN_POOLINGS_REAL + LEN_CONVS_REAL
                ]

                if get_num_convs(encoder_convs, "real") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el encoder de la "
                        f"capa real {capa} supera el máximo del cromosoma"
                    )

                if get_num_convs(decoder_convs, "real") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el decoder de la "
                        f"capa real {capa} supera el máximo del cromosoma"
                    )

            bottleneck = self.__real[-LEN_CONVS_REAL:]

            if get_num_convs(bottleneck, "real") > self.max_convs_per_layer:
                raise ValueError(
                    "El número de convoluciones en el bottleneck "
                    "real supera el máximo del cromosoma"
                )
        # Si self.binary está definido
        if self.__binary:
            if not isinstance(self.__binary, str):
                raise ValueError(
                    "El cromosoma binario no es un string"
                )

            if len(self.__binary) != LEN_CHROMOSOME_BIN:
                raise ValueError(
                    "El cromosoma binario no tiene el tamaño correcto"
                )

            if not all(v in ['0', '1'] for v in self.__binary):
                raise ValueError(
                    "Los valores del cromosoma binario no son todos 0 o 1"
                )

            num_layers_bin = get_num_layers(self.__binary, "binary")

            if num_layers is None:
                num_layers = num_layers_bin

            if num_layers_bin > self.max_layers:
                raise ValueError(
                    "El número de capas supera al máximo del cromosoma"
                )

            if num_layers_bin != num_layers:
                raise ValueError(
                    "Las capas no coinciden entre el cromosoma binario y el número de capas"
                )

            for i in range(0, len(self.__binary) - LEN_CONVS_BIN, LEN_LAYER_BIN):
                capa = i // LEN_LAYER_BIN
                encoder_convs = self.__binary[i:i + LEN_CONVS_BIN]
                decoder_convs = self.__binary[
                    i + LEN_CONVS_BIN + LEN_POOLINGS_BIN:
                    i + LEN_CONVS_BIN + LEN_POOLINGS_BIN + LEN_CONVS_BIN
                ]

                if get_num_convs(encoder_convs, "binary") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el encoder de la "
                        f"capa binaria {capa} supera el máximo del cromosoma"
                    )

                if get_num_convs(decoder_convs, "binary") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el decoder de la "
                        f"capa binaria {capa} supera el máximo del cromosoma"
                    )

            bottleneck = self.__binary[-LEN_CONVS_BIN:]

            if get_num_convs(bottleneck, "binary") > self.max_convs_per_layer:
                raise ValueError(
                    "El número de convoluciones en el bottleneck "
                    "binario supera el máximo del cromosoma"
                )
        if self.__decoded:
            if not isinstance(self.__decoded, tuple):
                raise ValueError(
                    "El cromosoma decodificado no es una tupla"
                )

            if len(self.__decoded) != 2:
                raise ValueError(
                    "El cromosoma decodificado no tiene el tamaño correcto"
                )

            if len(self.__decoded[0]) > MAX_LAYERS:
                raise ValueError(
                    "El número de capas supera al máximo permitido"
                )

            num_layers_decoded = get_num_layers(self.__decoded, "decoded")

            if num_layers is None:
                num_layers = num_layers_decoded

            if num_layers_decoded > self.max_layers:
                raise ValueError(
                    "El número de capas supera al máximo del cromosoma"
                )

            if num_layers_decoded != num_layers:
                raise ValueError(
                    "Las capas no coinciden entre el cromosoma decodificado y el número de capas"
                )

            for i, layer in enumerate(self.__decoded[0]):
                if layer is None:
                    continue

                encoder_convs, _ = layer[0]
                decoder_convs, _ = layer[1]

                if get_num_convs(encoder_convs, "decoded") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el encoder de la "
                        f"capa decodificada {i} supera el máximo del cromosoma"
                    )

                if get_num_convs(decoder_convs, "decoded") > self.max_convs_per_layer:
                    raise ValueError(
                        "El número de convoluciones en el decoder de la "
                        f"capa decodificada {i} supera el máximo del cromosoma"
                    )

            bottleneck = self.__decoded[1]

            if get_num_convs(bottleneck, "decoded") > self.max_convs_per_layer:
                raise ValueError(
                    "El número de convoluciones en el bottleneck "
                    "decodificado supera el máximo del cromosoma"
                )

    def generate_random(self, seed: Optional[int] = None):
        """
        Genera un cromosoma decodificado aleatorio

        Parameters
        ----------
        seed : Optional[int], optional
            Semilla para la generación de números aleatorios, by default `None`
        """
        # Limpiamos las caches caducadas
        self.data_loader = "0"
        self.data_loader_args = {}
        self.aptitude = None
        self.__decoded = ()
        self.__unet = None
        self.__real = []
        self.__binary = str()

        if seed is not None:
            random.seed(seed)
        elif self.seed is not None:
            random.seed(self.seed)

        active_layers = random.randint(2, self.max_layers)
        layers = [None] * (MAX_LAYERS - active_layers)

        for _ in range(active_layers):
            active_enc_convs = random.randint(1, self.max_convs_per_layer)
            encoder_convs = (
                [None] * (MAX_CONVS_PER_LAYER - active_enc_convs)
            )

            for _ in range(active_enc_convs):
                conv = (
                    random.choice(list(VALID_FILTERS.values())),
                    random.choice(list(KERNEL_SIZES.values())),
                    random.choice(list(ACTIVATION_FUNCTIONS.values()))
                )
                encoder_convs.append(conv)

            active_dec_convs = random.randint(1, self.max_convs_per_layer)
            decoder_convs = (
                [None] * (MAX_CONVS_PER_LAYER - active_dec_convs)
            )

            for _ in range(active_dec_convs):
                deconv = (
                    random.choice(list(VALID_FILTERS.values())),
                    random.choice(list(KERNEL_SIZES.values())),
                    random.choice(list(ACTIVATION_FUNCTIONS.values()))
                )
                decoder_convs.append(deconv)

            pooling = random.choice(list(VALID_POOLINGS.values()))
            concat = random.choice(list(CONCATENATION.values()))

            layers.append(((encoder_convs, pooling), (decoder_convs, concat)))

        active_btnk_convs = random.randint(1, self.max_convs_per_layer)
        bottleneck = [None] * (MAX_CONVS_PER_LAYER - active_btnk_convs)

        for _ in range(active_btnk_convs):
            conv = (
                random.choice(list(VALID_FILTERS.values())),
                random.choice(list(KERNEL_SIZES.values())),
                random.choice(list(ACTIVATION_FUNCTIONS.values()))
            )

            bottleneck.append(conv)

        self.__decoded = (layers, bottleneck)
        self.validate()

    # ========================
    # NOTE: Setters
    # ========================
    def set_decoded(self, decoded_chromosome: Optional[tuple] = None, **kwargs: int):
        """
        Crea el cromosoma decodificado a partir de otro o a partir de los cromosomas real o binario

        Parameters
        ----------
        decoded_chromosome : Optional[tuple], optional
            (Cromosoma decodificado, número de capas), by default `None`
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios
        """
        if decoded_chromosome:
            # === Si recibimos un cromosoma decodificado lo asignamos ===
            self.__decoded = decoded_chromosome
            # Limpiamos las caches caducadas, se volverán a generar cuando se necesiten
            self.data_loader = "0"
            self.data_loader_args = {}
            self.aptitude = None
            self.__unet = None
            self.__real = []
            self.__binary = str()
            self.validate()

            return

        # === Si no recibimos un cromosoma decodificado lo generamos ===
        # Primero intenta decodificar a través de la codificación real
        if self.__real:
            chromosome = self.__real
            real = True
        elif self.__binary:
            chromosome = self.__binary
            real = False
        else:
            self.generate_random(**kwargs)

            return

        self.__decoded = decode_chromosome(
            chromosome=chromosome,
            real=real
        )
        self.validate()

    def set_real(self, real_chromosome: Optional[list[float]] = None, **kwargs: int):
        """
        Crea el cromosoma real a partir de otro o a partir del cromosoma decodificado

        Parameters
        ----------
        real_chromosome : Optional[tuple], optional
            (Cromosoma real, número de capas), by default `None`
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios
        """
        if real_chromosome:
            # === Si recibimos un cromosoma real lo asignamos ===
            self.__real = real_chromosome
            # Limpiamos las caches caducadas
            self.data_loader = "0"
            self.data_loader_args = {}
            self.aptitude = None
            self.__decoded = ()
            self.__unet = None
            self.__binary = str()
            self.validate()
            # Creamos el cromosoma decodificado, el binario se creará cuando se necesite
            self.set_decoded()

            return

        # === Si no recibimos un cromosoma real lo generamos ===
        self.__real = []

        # Si el cromosoma no está definido, lo generamos
        if not self.__decoded:
            self.set_decoded(**kwargs)

        self.__real: list[float] = encode_chromosome(
            self.__decoded,
            real=True
        )
        self.validate()

    def set_binary(self, binary_chromosome: Optional[str] = None, **kwargs: int):
        """
        Crea el cromosoma binario a partir de otro o a partir del cromosoma decodificado

        Parameters
        ----------
        binary_chromosome : Optional[tuple], optional
            (Cromosoma binario, número de capas), by default `None`
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios
        """
        if binary_chromosome:
            # === Si recibimos un cromosoma binario lo asignamos ===
            binary = binary_chromosome
            self.__binary = unzip_binary(binary)
            # Limpiamos las caches caducadas
            self.data_loader = "0"
            self.data_loader_args = {}
            self.aptitude = None
            self.__decoded = ()
            self.__unet = None
            self.__real = []
            self.validate()
            # Creamos el cromosoma decodificado, el real se creará cuando se necesite
            self.set_decoded()

            return

        # === Si no recibimos un cromosoma binario lo generamos ===
        self.__binary = str()

        # Si el cromosoma no está definido, lo generamos
        if not self.__decoded:
            self.set_decoded(**kwargs)

        self.__binary: str = encode_chromosome(
            self.__decoded,
            real=False
        )
        self.validate()

    def set_unet(self, **kwargs: int):
        """
        Crea el modelo UNet a partir del cromosoma decodificado

        Parameters
        ----------
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

            Argumentos adicionales para el modelo UNet
            - in_channels : (int) Número de canales de entrada
        """
        # Limpiamos las caches caducadas (solo relacionadas con el modelo UNet)
        self.data_loader = "0"
        self.data_loader_args = {}
        self.aptitude = None
        self.__unet = None

        # Si el cromosoma no está definido, lo generamos
        if not self.__decoded:
            seed = kwargs.pop("seed", None)

            self.set_decoded(seed=seed)

        self.__unet = UNet(
            decoded_chromosome=self.__decoded,
            **kwargs
        )

    def set_aptitude(self, data_loader: Optional[Union[TorchDataLoader, str]] = None,
                     metric: str = "iou", **kwargs: Union[str, int, float]):
        """
        Evalúa el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con las imágenes a evaluar, si no se proporciona se utiliza el DataLoader
            con el que se entrenó el modelo, by default `None`

            Opciones (si se proporciona un string):
                - "coco-people"
                - "coco-car"
                - "carvana"
                - "road"
                - "car"
        metric : str, optional
            Métrica a utilizar para calcular la pérdida, by default `"iou"`

            Opciones:
                - "iou"
                - "dice"
                - "dice crossentropy"
        **kwargs : T.Compose or str or int or float
            Argumentos adicionales para el DataLoader:
            - batch_size : (int) Tamaño del batch
            - train_val_prop : (float) Proporción que se usará entre train y validation

            Argumentos adicionales para el Dataset:
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - img_width : (int) Ancho a redimensionar las imágenes
            - img_height : (int) Alto a redimensionar las imágenes

        Returns
        -------
        float
            Pérdida del modelo
        """
        if not self.__unet:
            self.set_unet()

        if not data_loader:
            if not self.data_loader:
                raise ValueError("No se ha especificado un DataLoader")

            if not kwargs:
                kwargs = self.data_loader_args

            data_loader = TorchDataLoader(
                self.data_loader,
                **kwargs
            )
        elif isinstance(data_loader, str):
            data_loader = TorchDataLoader(
                data_loader,
                **kwargs
            )

        imgs, masks = next(iter(data_loader.validation))
        imgs = imgs.to(CUDA)
        self.__unet = self.__unet.to(CUDA)
        self.__unet.eval()

        with autocast(device_type="cuda", dtype=float16):
            output = self.__unet(imgs)

        self.aptitude = eval_model(
            scores=output.float(),
            target=masks.float(),
            metrics=[metric],
            # Loss: True para minimización
            loss=True,
            items=True
        )[0]

    # ========================
    # NOTE: Getters
    # ========================
    def get_decoded(self, **kwargs: int) -> tuple[list[tuple[tuple[list[tuple[int, int, str]],
                                                                   str],
                                                             tuple[list[tuple[int, int, str]],
                                                                   bool]]],
                                                  list[tuple[int, int, str]]]:
        """
        Devuelve el estado actual del cromosoma sin codificación

        Parameters
        ----------
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

        Returns
        -------
        tuple
            Cromosoma decodificado
        """
        if not self.__decoded:
            self.set_decoded(**kwargs)

        return self.__decoded

    def get_real(self, **kwargs: int) -> list[float]:
        """
        Devuelve el estado actual del cromosoma con codificación real

        Parameters
        ----------
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

        Returns
        -------
        list
            Cromosoma en su representación real
        """
        if not self.__real:
            self.set_real(**kwargs)

        return self.__real

    def get_binary(self, zip: bool = False, **kwargs: int) -> str:
        """
        Devuelve el estado actual del cromosoma con codificación binaria

        Parameters
        ----------
        zip : bool, optional
            Si se devolverá el cromosoma binario comprimido, by default `False`
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

        Returns
        -------
        str
            Cromosoma en su representación binaria
        """
        if not self.__binary:
            self.set_binary(**kwargs)

        if zip:
            return zip_binary(self.__binary)

        return self.__binary

    def get_unet(self, **kwargs: int) -> UNet:
        """
        Devuelve el modelo UNet

        Parameters
        ----------
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

            Argumentos adicionales para el modelo UNet
            - in_channels : (int) Número de canales de entrada

        Returns
        -------
        UNet
            Modelo UNet
        """
        if not self.__unet:
            self.set_unet(**kwargs)

        return self.__unet

    def get_aptitude(self, **kwargs: Union[str, int, float, TorchDataLoader]) -> float:
        """
        Devuelve la aptitud del cromosoma

        Parameters
        ----------
        **kwargs : T.Compose or str or int or float
            Argumentos adicionales para la evaluación del modelo:
            - data_loader : (TorchDataLoader or str) DataLoader con las imágenes a evaluar,
                            si no se proporciona se utiliza el DataLoader con el que se entrenó el modelo
                            ("coco-people", "coco-car", "carvana", "road", "car")
            - metric : (str) Métrica a utilizar para calcular la pérdida
                       ("iou", "dice" o "dice crossentropy")

            Argumentos adicionales para el DataLoader:
            - batch_size : (int) Tamaño del batch
            - train_val_prop : (float) Proporción que se usará entre train y validation

            Argumentos adicionales para el Dataset:
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - img_width : (int) Ancho a redimensionar las imágenes
            - img_height : (int) Alto a redimensionar las imágenes

        Returns
        -------
        float
            Aptitud del cromosoma
        """
        if (self.aptitude is None
                or kwargs.get("data_loader", None) is not None
                or any(arg in kwargs for arg in TorchDataLoader.ARGS)):
            self.set_aptitude(**kwargs)

        return self.aptitude

    def get_num_layers(self, **kwargs: int) -> int:
        """
        Devuelve el número de capas del cromosoma

        Parameters
        ----------
        **kwargs : int
            Argumentos adicionales para la generación del cromosoma si se debe generar uno nuevo:
            - seed : (int) Semilla para la generación de números aleatorios

        Returns
        -------
        int
            Número de capas del cromosoma
        """
        if not self.__decoded:
            self.set_decoded(**kwargs)

        return get_num_layers(self.__decoded, "decoded")

    # ========================
    # NOTE: Funciones de la UNet
    # ========================
    def train_unet(self, data_loader: Union[TorchDataLoader, str],
                   **kwargs: Union[str, int, float, bool]) -> tuple[float,
                                                                    int,
                                                                    dict[str,
                                                                         list[float]]]:
        """
        Entrena el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con los datos de entrenamiento y validación

            Opciones (si se proporciona un string):
                - "coco-people"
                - "coco-car"
                - "carvana"
                - "road"
                - "car"
        **kwargs : T.Compose or str or int or float or bool
            Argumentos adicionales para el entrenamiento:
            - metric : (str) Métrica a utilizar para calcular la pérdida
                       ("iou", "dice" o "dice crossentropy")
            - lr : (float) Tasa de aprendizaje
            - epochs : (int) Número de épocas
            - early_stopping : (bool) Si se debe usar el early stopping
            - early_stopping_patience : (int) Número de épocas a esperar sin mejora antes de
                                        detener el entrenamiento
            - early_stopping_delta : (float) Umbral mínimo de mejora para considerar un progreso
            - stopping_threshold : (float) Umbral de rendimiento para la métrica de validación.
                                   Si se alcanza o supera, el entrenamiento se detiene
            - infinite : (bool) Si el entrenamiento es infinito
            - show_val : (bool) Si mostrar los resultados de la validación en cada epoch

            Argumentos adicionales para el DataLoader:
            - k_folds_subsets : (tuple) Subsets para k_folds validation
            - batch_size : (int) Tamaño del batch
            - train_val_prop : (float) Proporción que se usará entre train y validation

            Argumentos adicionales para el Dataset:
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - img_width : (int) Ancho a redimensionar las imágenes
            - img_height : (int) Alto a redimensionar las imágenes

        Returns
        -------
        tuple
            (Tiempo de entrenamiento en segundos, última época, resultados de las métricas
            a lo largo del entrenamiento)
        """
        if not self.__unet:
            self.set_unet()

        self.data_loader_args, kwargs = TorchDataLoader.get_args(kwargs)

        if isinstance(data_loader, str):
            data_loader = TorchDataLoader(
                data_loader,
                **self.data_loader_args
            )

        self.data_loader = str(data_loader)
        set_current_net_binary(self.get_binary(zip=True))

        start = time.perf_counter()

        self.__unet, last_epoch, metrics = train_model(
            model=self.__unet,
            data_loader=data_loader,
            **kwargs
        )

        time_seconds = time.perf_counter() - start

        print(
            "Entrenamiento finalizado en "
            f"{time_seconds:.4f} segundos"
        )

        return time_seconds, last_epoch, metrics

    def show_results(self, data_loader: Optional[Union[TorchDataLoader, str]] = None,
                     name: Optional[str] = None, **kwargs: Union[str, int, float, bool]):
        """
        Muestra los resultados del modelo UNet actual con el conjunto de test

        Parameters
        ----------
        data_loader : TorchDataLoader or str, optional
            DataLoader con las imágenes a evaluar, si no se especifica se usará el DataLoader
            con el que se entrenó, by default `None`

            Opciones (si se proporciona un string):
                - "coco-people"
                - "coco-car"
                - "carvana"
                - "road"
                - "car"
        name : Optional[str], optional
            Nombre del archivo, si no se especifica el nombre será el hash del cromosoma
            binario, by default `None`
        **kwargs : T.Compose or str or int or float or bool
            Argumentos adicionales para la generación de la gráfica:
            - save : (bool) Si se guardan las imágenes o se muestran
            - show_size : (int) Número de imágenes a mostrar
            - path : (str) Ruta donde se guardarán las imágenes

            Argumentos adicionales para el DataLoader:
            - batch_size : (int) Tamaño del batch
            - train_val_prop : (float) Proporción que se usará entre train y validation

            Argumentos adicionales para el Dataset:
            - data_path : (str) Ruta de los datos
            - dataset_len : (int) Número de imágenes a cargar
            - test_prop : (float) Proporción de imágenes que se usará entre el conjunto de
                          entrenamiento (train y validation) y test
            - img_width : (int) Ancho a redimensionar las imágenes
            - img_height : (int) Alto a redimensionar las imágenes
        """
        if not self.__unet:
            self.set_unet()

        data_loader_args, kwargs = TorchDataLoader.get_args(kwargs)

        if not data_loader:
            if not self.data_loader:
                raise ValueError("No se ha especificado un DataLoader")

            if not data_loader_args:
                data_loader_args = self.data_loader_args

            data_loader = TorchDataLoader(
                self.data_loader,
                **data_loader_args
            )
        elif isinstance(data_loader, str):
            data_loader = TorchDataLoader(
                data_loader,
                **data_loader_args
            )

        if name is None:
            name = self.__str__() + ".png"
            name = name.replace("/", "#s").replace("\\", "#b")

        plot_results(
            model=self.__unet,
            test_loader=data_loader.test,
            name=name,
            **kwargs
        )

    def save_unet(self, name: Optional[str] = None, **kwargs: str):
        """
        Guarda el modelo UNet en un archivo

        Parameters
        ----------
        name : Optional[str], optional
            Nombre del archivo, si no se especifica el nombre será el hash del cromosoma
            binario, by default `None`

        **kwargs : str
            Argumentos adicionales para el guardado del modelo:
            - path : (str) Ruta donde se guardará el modelo
        """
        if not self.__unet:
            self.set_unet()

        if name is None:
            name = self.__str__() + ".pt"
            name = name.replace("/", "#s").replace("\\", "#b")

        save_model(
            model=self.__unet,
            name=name,
            **kwargs
        )

    def remove_checkpoints(self):
        """
        Elimina los checkpoints del modelo UNet
        """
        remove_checkpoints(self.get_binary(zip=True))
