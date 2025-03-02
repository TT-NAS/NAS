import time
import random
from typing import Union, Optional

from torch_utils import CUDA
from torch_utils import UNet, TorchDataLoader
from torch_utils import plot_results, train_model, save_model, eval_model
from .encode import zip_binary, encode_chromosome
from .decode import unzip_binary, decode_chromosome
from .constants import FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS, POOLINGS, CONCATENATION


class Chromosome:
    """
    Clase que representa un cromosoma de una red UNet
    """

    def __init__(self, max_layers: int, max_conv_per_layer: int, seed: Optional[int] = None, chromosome: Optional[Union[tuple, list, str]] = None):
        """
        Clase que representa un cromosoma de una red UNet

        A tener en cuenta:
        - El numero de convoluciones en el cuello de botella es el mismo que el max_conv_per_layer

        Parameters
        ----------
        max_layers : int
            Número máximo de capas de la red
        max_conv_per_layer : int
            Número máximo de capas convolucionales por bloque
        seed : Optional[int], optional
            Semilla para la generación de números aleatorios, by default None
        chromosome : Optional[tuple or list or str], optional
            Cromosoma de un individuo a cargar, by default None
        """
        self.max_layers = max_layers
        self.max_conv_per_layer = max_conv_per_layer
        self.seed = seed

        # __decoded será el cromosoma como tal, __real y __binary son solo auxiliares y caché
        self.data_loader = "0"
        self.data_loader_args = {}
        self.aptitude = None
        self.__decoded = ()
        self.__unet = None
        self.__real = []
        self.__binary = str()
        self.num_layers = None

        # 3 valores por convolución en encoding y decoding + 2 valores por pooling y concatenación
        self.REAL_LAYER_LEN = 3 * self.max_conv_per_layer * 2 + 2
        self.REAL_BOTTLENECK_LEN = 3 * self.max_conv_per_layer
        # filters: 4 bits, kernel_size: 2 bits, activation: 4 bits = 10 bits
        # pooling: 1 bit, concat: 1 bit = 2 bits
        self.BIN_LAYER_LEN = 10 * self.max_conv_per_layer * 2 + 2
        self.BIN_BOTTLENECK_LEN = 10 * self.max_conv_per_layer

        if chromosome:
            if isinstance(chromosome, tuple):
                self.__decoded = chromosome
            elif isinstance(chromosome, list):
                self.__real = chromosome
            elif isinstance(chromosome, str):
                self.__binary = unzip_binary(chromosome)

            self.validate()

    # ========================
    # NOTE: Utils de la clase
    # ========================
    def __str__(self):
        transform = repr(self.data_loader_args.get(
            "transform", "0"
        )).replace("\n", "").replace(" ", "").replace("'", "")
        data_path = self.data_loader_args.get(
            "data_path", "0"
        ).replace("\n", "")

        return f"{self.get_binary(zip=True)}-dl{self.data_loader}-t{transform}-dp{data_path}"

    def __repr__(self):
        return f"Chromosome(max_layers={self.max_layers}, max_conv_per_layer={self.max_conv_per_layer}, seed={self.seed}, chromosome='{self.get_binary(zip=True)}')"

    def validate(self):
        """
        Valida que el cromosoma tenga un tamaño correcto

        Raises
        -------
        ValueError
            Si el cromosoma no cumple con las condiciones necesarias
        """
        num_layers_real = num_layers_bin = None

        # Si self.real está definido
        if self.__real:
            if not isinstance(self.__real, list):
                raise ValueError(
                    "El cromosoma real no es una lista"
                )

            num_layers_real = (
                (len(self.__real) - self.REAL_BOTTLENECK_LEN)
                // self.REAL_LAYER_LEN
            )

            if self.num_layers is None:
                self.num_layers = num_layers_real
                print("num_layers ajustado según __real")

            if self.num_layers != num_layers_real:
                raise ValueError(
                    "Las capas no coinciden entre el cromosoma real y el número de capas"
                )

            total_params = self.num_layers * self.REAL_LAYER_LEN + self.REAL_BOTTLENECK_LEN

            if len(self.__real) != total_params:
                raise ValueError(
                    "El cromosoma real no tiene el tamaño correcto"
                )

            if not all(0 <= v <= 1 for v in self.__real):
                raise ValueError(
                    "Los valores del cromosoma real se salen del rango [0, 1]"
                )

        # Si self.binary está definido
        if self.__binary:
            if not isinstance(self.__binary, str):
                raise ValueError(
                    "El cromosoma binario no es un string"
                )

            num_layers_bin = (
                (len(self.__binary) - self.BIN_BOTTLENECK_LEN)
                // self.BIN_LAYER_LEN
            )

            if self.num_layers is None:
                self.num_layers = num_layers_bin
                print("num_layers ajustado según __binary")

            if self.num_layers != num_layers_bin:
                raise ValueError(
                    "Las capas no coinciden entre el cromosoma binario y el número de capas"
                )

            total_params = self.num_layers * self.BIN_LAYER_LEN + self.BIN_BOTTLENECK_LEN

            if len(self.__binary) != total_params:
                raise ValueError(
                    "El cromosoma binario no tiene el tamaño correcto"
                )

            if not all(v in ['0', '1'] for v in self.__binary):
                raise ValueError(
                    "Los valores del cromosoma binario no son 0 o 1"
                )

        if self.__decoded:
            if self.num_layers is None:
                self.num_layers = len(self.__decoded[0])
                print("num_layers ajustado según __decoded")

            if self.num_layers != len(self.__decoded[0]):
                raise ValueError(
                    "Las capas no coinciden entre el cromosoma decodificado y el número de capas"
                )

            if len(self.__decoded[1]) != self.max_conv_per_layer or len(self.__decoded[0][0][0][0]) != self.max_conv_per_layer:
                raise ValueError(
                    "El espacio para convoluciones no es compatible con el número máximo permitido"
                )
            # TODO: Crear función para contar convoluciones activas dentro de una capa y validar con max_conv_per_layer

        if self.num_layers is not None and self.num_layers > self.max_layers:
            raise ValueError("El número de capas supera el máximo permitido")

    def generate_random(self, seed: Optional[int] = None):
        """
        Genera un cromosoma decodificado aleatorio

        Parameters
        ----------
        seed : Optional[int], optional
            Semilla para la generación de números aleatorios, by default None
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
            random.seed(
                int(
                    str(seed) +
                    str(self.max_layers) +
                    str(self.max_conv_per_layer)
                )
            )
        elif self.seed is not None:
            random.seed(
                int(
                    str(self.seed) +
                    str(self.max_layers) +
                    str(self.max_conv_per_layer)
                )
            )

        self.num_layers = random.randint(2, self.max_layers)

        layers = []

        for _ in range(self.num_layers):
            encoder_convs = []
            decoder_convs = []
            pooling = random.choice(list(POOLINGS.values()))
            concat = random.choice(list(CONCATENATION.values()))

            for _ in range(self.max_conv_per_layer):
                conv = (
                    random.choice(list(FILTERS.values())),
                    random.choice(list(KERNEL_SIZES.values())),
                    random.choice(list(ACTIVATION_FUNCTIONS.values()))
                )
                deconv = (
                    random.choice(list(FILTERS.values())),
                    random.choice(list(KERNEL_SIZES.values())),
                    random.choice(list(ACTIVATION_FUNCTIONS.values()))
                )
                encoder_convs.append(conv)
                decoder_convs.append(deconv)

            layers.append(((encoder_convs, pooling), (decoder_convs, concat)))

        bottleneck = []

        for _ in range(self.max_conv_per_layer):
            conv = (
                random.choice(list(FILTERS.values())),
                random.choice(list(KERNEL_SIZES.values())),
                random.choice(list(ACTIVATION_FUNCTIONS.values()))
            )

            bottleneck.append(conv)

        self.__decoded = (layers, bottleneck)
        self.validate()

    # ========================
    # NOTE: Setters
    # ========================
    def set_decoded(self, decoded_chromosome: Optional[tuple[tuple, int]] = None):
        """
        Crea el cromosoma decodificado a partir de otro o a partir de los cromosomas real o binario

        Parameters
        ----------
        decoded_chromosome : Optional[tuple], optional
            (Cromosoma decodificado, Número de capas), by default None
        """
        if decoded_chromosome:
            # === Si recibimos un cromosoma decodificado lo asignamos ===
            self.__decoded, self.num_layers = decoded_chromosome
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
            layer_len = self.REAL_LAYER_LEN
            bottleneck_len = self.REAL_BOTTLENECK_LEN
            real = True
        elif self.__binary:
            chromosome = self.__binary
            layer_len = self.BIN_LAYER_LEN
            bottleneck_len = self.BIN_BOTTLENECK_LEN
            real = False
        else:
            self.generate_random()

            return

        self.__decoded = decode_chromosome(
            chromosome=chromosome,
            layer_len=layer_len,
            bottleneck_len=bottleneck_len,
            real=real
        )
        self.validate()

    def set_real(self, real_chromosome: Optional[tuple[list[float], int]] = None):
        """
        Crea el cromosoma real a partir de otro o a partir del cromosoma decodificado

        Parameters
        ----------
        real_chromosome : Optional[tuple], optional
            (Cromosoma real, Número de capas), by default None
        """
        if real_chromosome:
            # === Si recibimos un cromosoma real lo asignamos ===
            self.__real, self.num_layers = real_chromosome
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
            self.set_decoded()

        self.__real: list[float] = encode_chromosome(
            self.__decoded,
            real=True
        )
        self.validate()

    def set_binary(self, binary_chromosome: Optional[tuple[str, int]] = None):
        """
        Crea el cromosoma binario a partir de otro o a partir del cromosoma decodificado

        Parameters
        ----------
        binary_chromosome : Optional[tuple], optional
            (Cromosoma binario, Número de capas), by default None
        """
        if binary_chromosome:
            # === Si recibimos un cromosoma binario lo asignamos ===
            binary, self.num_layers = binary_chromosome
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
            self.set_decoded()

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
            Argumentos adicionales para el modelo UNet
            - in_channels : (int) Número de canales de entrada
        """
        self.data_loader = "0"
        self.data_loader_args = {}
        self.aptitude = None
        self.__unet = None

        # Si el cromosoma no está definido, lo generamos
        if not self.__decoded:
            self.set_decoded()

        self.__unet = UNet(
            decoded_chromosome=self.__decoded,
            **kwargs
        )

    def set_aptitude(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, metric: str = "iou", **kwargs: Union[float, str, object]):
        """
        Evalúa el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con las imágenes a evaluar, si no se proporciona se utiliza el DataLoader con el que se entrenó el modelo, by default None
        metric : str, optional
            Métrica a utilizar para calcular la pérdida, by default "iou"

            Opciones:
                - "iou"
                - "dice"
                - "dice crossentropy"
                - "accuracy"
        **kwargs : float or T.Compose or str
            Argumentos adicionales para las funciones de pérdida:
            - smooth : (float) Valor para suavizar la división
            - threshold : (float) Umbral para considerar si un píxel es 1 o 0

            Argumentos adicionales para el dataset:
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes
            - data_path : (str) Ruta de los datos

        Returns
        -------
        float
            Pérdida del modelo
        """
        if not self.__unet:
            self.set_unet()

        data_loader_args = None
        if "transform" in kwargs or "data_path" in kwargs:
            data_loader_args = {
                k: v for k, v in kwargs.items()
                if k in ["transform", "data_path"]
            }
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["transform", "data_path"]
            }

        if not data_loader:
            if not self.data_loader:
                raise ValueError("No se ha especificado un DataLoader")

            if data_loader_args is not None:
                self.data_loader_args = data_loader_args

            data_loader = TorchDataLoader(
                self.data_loader,
                **self.data_loader_args
            )
        elif isinstance(data_loader, str):
            if data_loader_args is None:
                data_loader_args = {}

            data_loader = TorchDataLoader(
                data_loader,
                **data_loader_args
            )

        imgs = next(iter(data_loader.test))
        imgs = imgs.to(CUDA)
        self.__unet = self.__unet.to(CUDA)

        self.__unet.eval()
        output = self.__unet(imgs)

        self.aptitude = eval_model(
            scores=output,
            target=imgs,
            metrics=[metric],
            # Loss: True para minimización
            loss=True,
            items=True,
            **kwargs
        )[0]

    # ========================
    # NOTE: Getters
    # ========================
    def get_decoded(self) -> tuple[list[tuple[tuple[list[tuple[int, int, str]],
                                                    str],
                                              tuple[list[tuple[int, int, str]],
                                                    bool]]],
                                   list[tuple[int, int, str]]]:
        """
        Devuelve el estado actual del cromosoma sin codificación

        Returns
        -------
        tuple
            Cromosoma decodificado
        """
        if not self.__decoded:
            self.set_decoded()

        return self.__decoded

    def get_real(self) -> list[float]:
        """
        Devuelve el estado actual del cromosoma con codificación real

        Returns
        -------
        list
            Cromosoma en su representación real
        """
        if not self.__real:
            self.set_real()

        return self.__real

    def get_binary(self, zip: bool = False) -> str:
        """
        Devuelve el estado actual del cromosoma con codificación binaria

        Returns
        -------
        str
            Cromosoma en su representación binaria
        """
        if not self.__binary:
            self.set_binary()

        if zip:
            return zip_binary(self.__binary)

        return self.__binary

    def get_unet(self) -> UNet:
        """
        Devuelve el modelo UNet

        Returns
        -------
        UNet
            Modelo UNet
        """
        if not self.__unet:
            self.set_unet()

        return self.__unet

    def get_aptitude(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, metric: str = "iou", **kwargs: Union[float, str, object]) -> float:
        """
        Devuelve la aptitud del cromosoma

        Parameters
        ----------
        data_loader : TorchDataLoader or str, optional
            DataLoader con las imágenes a evaluar, by default None
        metric : str, optional
            Métrica a utilizar para calcular la pérdida, by default "iou"
        **kwargs : float or T.Compose or str
            Argumentos adicionales para las funciones de pérdida:
            - smooth : (float) Valor para suavizar la división
            - threshold : (float) Umbral para considerar si un píxel es 1 o 0

            Argumentos adicionales para el dataset:
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes
            - data_path : (str) Ruta de los datos

        Returns
        -------
        float
            Aptitud del cromosoma
        """
        if self.aptitude is None or data_loader or "transform" in kwargs or "data_path" in kwargs:
            self.set_aptitude(data_loader, metric, **kwargs)

        return self.aptitude

    # ========================
    # NOTE: Funciones de la UNet
    # ========================
    def train_unet(self, data_loader: Union[TorchDataLoader, str], **kwargs: Union[str, float, int, bool, object]):
        """
        Entrena el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con los datos de entrenamiento y validación
        **kwargs : str or int or float or bool or T.Compose
            Argumentos adicionales para el entrenamiento:
            - metric : (str) Métrica a utilizar para calcular la pérdida. ("iou", "dice", "dice crossentropy" o "accuracy")
            - lr : (float) Tasa de aprendizaje
            - epochs : (int) Número de épocas
            - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
            - print_every : (int) Cada cuántos pasos se imprime el resultado

            Argumentos adicionales para el dataset:
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes
            - data_path : (str) Ruta de los datos
        """
        if not self.__unet:
            self.set_unet()

        if "transform" in kwargs or "data_path" in kwargs:
            self.data_loader_args = {
                k: v for k, v in kwargs.items()
                if k in ["transform", "data_path"]
            }
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["transform", "data_path"]
            }

        if isinstance(data_loader, str):
            data_loader = TorchDataLoader(
                data_loader,
                **self.data_loader_args
            )

        self.data_loader = str(data_loader)

        start = time.time()

        train_model(
            model=self.__unet,
            data_loader=data_loader,
            **kwargs
        )

        print(
            "Entrenamiento finalizado en "
            f"{time.time() - start:.3f} segundos"
        )

    def show_results(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, save: bool = False, name: Optional[str] = None, **kwargs: Union[str, object]):
        """
        Muestra los resultados del modelo UNet actual

        Parameters
        ----------
        data_loader : TorchDataLoader or str, optional
            DataLoader con las imágenes a evaluar, si no se especifica se usará el DataLoader con el que se entrenó, by default None
        **kwargs : str or T.Compose
            Argumentos adicionales para el dataset:
            - transform : (T.Compose) Transformaciones a aplicar a las imágenes
            - data_path : (str) Ruta de los datos
        """
        if not self.__unet:
            self.set_unet()

        data_loader_args = None
        if "transform" in kwargs or "data_path" in kwargs:
            data_loader_args = {
                k: v for k, v in kwargs.items()
                if k in ["transform", "data_path"]
            }
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["transform", "data_path"]
            }

        if not data_loader:
            if not self.data_loader:
                raise ValueError("No se ha especificado un DataLoader")

            if data_loader_args is not None:
                self.data_loader_args = data_loader_args

            data_loader = TorchDataLoader(
                self.data_loader,
                **self.data_loader_args
            )
        elif isinstance(data_loader, str):
            if data_loader_args is None:
                data_loader_args = {}

            data_loader = TorchDataLoader(
                data_loader,
                **data_loader_args
            )

        if save and not name:
            name = self.__str__() + ".png"
            name = name.replace("/", "#s").replace("\\", "#b")

        plot_results(
            model=self.__unet,
            test_loader=data_loader.test,
            save=save,
            name=name
        )

    def save_unet(self, name: Optional[str] = None):
        """
        Guarda el modelo UNet en un archivo

        Parameters
        ----------
        name : Optional[str], optional
            Nombre del archivo, si no se especifica el nombre será el hash del cromosoma binario, by default None
        """
        if not self.__unet:
            self.set_unet()

        if not name:
            name = self.__str__() + ".pt"
            name = name.replace("/", "#s").replace("\\", "#b")

        save_model(
            model=self.__unet,
            name=name
        )
        print(f"Modelo guardado como {name}")
