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

Statics
-------
- FILTERS : Opciones de filtros para las convoluciones
- KERNEL_SIZES : Opciones de tamaños de kernel para las convoluciones
- ACTIVATION_FUNCTIONS : Opciones de funciones de activación para las convoluciones
- POOLINGS : Opciones de pooling
- CONCATENATION : Opciones de concatenación

Variables
----------
- max_layers : Número máximo de capas de la red
- max_conv_per_layer : Número máximo de capas convolucionales por bloque
- seed : Semilla para la generación de números aleatorios
    - Se utiliza cuando el cromosoma se genera aleatoriamente
- data_loader : Identificador del DataLoader con el que se ha evaluado el cromosoma
    - Es una letra que indica el DataLoader utilizado ('c' para carvana, 'r' para road y '0' para no especificado)
    - Se asigna solo al entrenar el modelo, después puede se utiliza para deducir el DataLoader a ocupar en métodos
      como `get_aptitude` o `show_results` si no se proporciona uno
- aptitude : Aptitud del cromosoma
    - Es la pérdida del modelo UNet evaluado con el DataLoader especificado o guardado en `data_loader`
- __decoded : Es el cromosoma en su forma más entendible, con listas y tuplas
- __real : Es el cromosoma en su forma real, con valores entre 0 y 1
- __binary : Es el cromosoma en su forma binaria, con 0s y 1s
- num_layers : Número de capas de la red
    - No es necesario especificarlo, se deduce al validar el cromosoma

# Métodos de la clase Chromosome:

Utils
-----
- __str__ : Representación en string de la clase
- __repr__ : Representación en string de la clase para el debugger
- zip_binary : Comprime un cromosoma binario
    - Se utiliza para guardar el cromosoma binario en un archivo sin que el nombre sea tan largo
- unzip_binary : Descomprime un cromosoma binario
    - Se utiliza para leer un cromosoma binario comprimido
- validate : Valida que el cromosoma cumpla con las condiciones mínimas
- generate_random : Genera un cromosoma decodificado aleatorio

Funciones de codificación
-------------------------
Funciones para codificar un cromosoma decodificado en su forma real o binaria

Funciones de decodificación
---------------------------
Funciones para decodificar un cromosoma real o binario en su forma decodificada

Setters
-------
Funciones para asignar el cromosoma a partir de otro, o para generarlos si no están definidos

Getters
-------
Funciones para obtener el cromosoma en su forma decodificada, real, binaria, el modelo UNet o la aptitud del cromosoma

Funciones de la UNet
--------------------
Funciones para crear, evaluar y entrenar el modelo UNet

"""
import time
import random
import base64
from typing import Union, Optional

from Torch_utils import UNet, TorchDataLoader, CUDA
from Torch_utils import plot_results, train_model, save_model, eval_model


class Chromosome:
    """
    Clase que representa un cromosoma de una red UNet
    """

    FILTERS = {
        # TODO: Volver a hacer la base con la codificación en la que 0000 es 2**0 y no 0 :c
        # '0000': 0,
        '0001': 2**0,  # 1
        '0011': 2**1,  # 2
        '0010': 2**2,  # 4
        '0110': 2**3,  # 8
        '0111': 2**4,  # 16
        '0101': 2**5,  # 32
        '0100': 2**6,  # 64
        '1100': 2**7,  # 128
        '1101': 2**8,  # 256
        '1111': 2**9,  # 512
        '1110': 2**10,  # 1024
    }
    KERNEL_SIZES = {
        '00': 1,
        '01': 3,
        '11': 5,
    }
    ACTIVATION_FUNCTIONS = {
        '0000': 'relu',
        '0001': 'sigmoid',
        '0011': 'tanh',
        '0010': 'softmax',
        '0110': 'softplus',
        '0111': 'softsign',
        '0101': 'selu',
        '0100': 'elu',
        '1100': 'exponential',
        '1101': 'linear',
    }
    POOLINGS = {
        '0': 'max',
        '1': 'average',
    }
    CONCATENATION = {
        '0': False,
        '1': True,
    }

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
                self.__binary = self.unzip_binary(chromosome)

            self.validate()

    # ========================
    # NOTE: Utils de la clase
    # ========================
    def __str__(self):
        return f"{self.get_binary(zip=True)}-{self.data_loader}"

    def __repr__(self):
        return f"Chromosome(max_layers={self.max_layers}, max_conv_per_layer={self.max_conv_per_layer}, seed={self.seed}, chromosome='{self.get_binary(zip=True)}')"

    def zip_binary(self, binary: str) -> str:
        length = len(binary)
        byte_data = int(binary, 2).to_bytes(
            (length + 7) // 8,
            byteorder='big'
        )

        return f"{base64.b32encode(byte_data).decode().rstrip('=')}:{length}"

    def unzip_binary(self, binary: str) -> str:
        """
        Descomprime un cromosoma binario
        """
        if ':' not in binary:
            return binary

        encoded_data, length = binary.split(":")
        length = int(length)
        padding = '=' * ((8 - len(encoded_data) % 8) % 8)
        byte_data = base64.b32decode(encoded_data + padding)

        return bin(int.from_bytes(byte_data, byteorder='big'))[2:].zfill(length)

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

        if self.num_layers is None:
            self.num_layers = random.randint(2, self.max_layers)

        layers = []

        for _ in range(self.num_layers):
            encoder_convs = []
            decoder_convs = []
            pooling = random.choice(list(self.POOLINGS.values()))
            concat = random.choice(list(self.CONCATENATION.values()))

            for _ in range(self.max_conv_per_layer):
                conv = (
                    random.choice(list(self.FILTERS.values())),
                    random.choice(list(self.KERNEL_SIZES.values())),
                    random.choice(list(self.ACTIVATION_FUNCTIONS.values()))
                )
                deconv = (
                    random.choice(list(self.FILTERS.values())),
                    random.choice(list(self.KERNEL_SIZES.values())),
                    random.choice(list(self.ACTIVATION_FUNCTIONS.values()))
                )
                encoder_convs.append(conv)
                decoder_convs.append(deconv)

            layers.append(((encoder_convs, pooling), (decoder_convs, concat)))

        bottleneck = []

        for _ in range(self.max_conv_per_layer):
            conv = (
                random.choice(list(self.FILTERS.values())),
                random.choice(list(self.KERNEL_SIZES.values())),
                random.choice(list(self.ACTIVATION_FUNCTIONS.values()))
            )

            bottleneck.append(conv)

        self.__decoded = (layers, bottleneck)
        self.validate()

    # ========================
    # NOTE: Funciones de codificación
    # ========================
    @staticmethod
    def encode_gene(value: Union[int, bool, str], options: dict, real: bool) -> Union[float, str]:
        """
        Devuelve la codificación equivalente a value en las opciones

        Parameters
        ----------
        value : int or bool or str
            Valor a codificar
        options : dict
            Diccionario de valores y su codificación
        real : bool
            Indica si la codificación deseada es real o binaria

        Returns
        -------
        float or str
            El valor codificado en real (dentro de un rango de [0, 1]) o binario (en un string de 0s y 1s)
            Valor codificado en
            - un flotante con rango [0, 1] si `real=True`
            - un string con 0s y 1s si `real=False`

        Raises
        -------
        ValueError
            Si el valor no está en las opciones
        """
        if real:
            step = 1 / len(options)
            real_representation = 0.01

            for _, v in options.items():
                if v == value:
                    return round(real_representation, 2)

                real_representation += step

        else:
            for k, v in options.items():
                if v == value:
                    return k

        raise ValueError(f"El valor {value} no está en las opciones")

    @classmethod
    def encode_convs(cls, convs: list[tuple[int, int, str]], real: bool) -> Union[list[float], str]:
        """
        Codifica una lista de convoluciones

        Parameters
        ----------
        fsa_blocks : list
            Lista con los valores de las convoluciones
        real : bool
            Indica si la codificación deseada es real o binaria

        Returns
        -------
        list or str
            Convoluciones codificadas en
            - una lista de flotantes si `real=True`
            - un string si `real=False`
        """
        encoded_convs = [] if real else ""

        for f, s, a in convs:
            encoded_conv = [
                Chromosome.encode_gene(
                    value=f,
                    options=cls.FILTERS,
                    real=real
                ),
                Chromosome.encode_gene(
                    value=s,
                    options=cls.KERNEL_SIZES,
                    real=real
                ),
                Chromosome.encode_gene(
                    value=a,
                    options=cls.ACTIVATION_FUNCTIONS,
                    real=real
                )
            ]

            if real:
                encoded_convs.extend(encoded_conv)
            else:
                encoded_conv = "".join(encoded_conv)
                encoded_convs += encoded_conv

        return encoded_convs

    @classmethod
    def encode_chromosome(
        cls,
        decoded_chromosome:
            tuple[list[tuple[tuple[list[tuple[int, int, str]],
                                   str],
                             tuple[list[tuple[int, int, str]],
                                   bool]]],
                  list[tuple[int, int, str]]],
        real: bool
    ) -> Union[list[float], str]:
        """
        Devuelve un cromosoma en codificación real o binaria

        Parameters
        ----------
        decoded_chromosome : tuple
            Cromosoma decodificado
        real : bool
            Indica si la codificación deseada es real o binaria

        Returns
        -------
        list or str
            Cromosoma codificado en
            - una lista de flotantes si `real=True`
            - un string si `real=False`
        """
        encoded_chromosome = [] if real else str()

        for encoder, decoder in decoded_chromosome[0]:
            encoded_encoder = cls.encode_convs(encoder[0], real=real)
            encoded_pooling = Chromosome.encode_gene(
                value=encoder[1],
                options=cls.POOLINGS,
                real=real
            )
            encoded_decoder = cls.encode_convs(decoder[0], real=real)
            encoded_concat = Chromosome.encode_gene(
                value=decoder[1],
                options=cls.CONCATENATION,
                real=real
            )

            if real:
                encoded_chromosome.extend(encoded_encoder)
                encoded_chromosome.append(encoded_pooling)
                encoded_chromosome.extend(encoded_decoder)
                encoded_chromosome.append(encoded_concat)

            else:
                encoded_chromosome += encoded_encoder
                encoded_chromosome += encoded_pooling
                encoded_chromosome += encoded_decoder
                encoded_chromosome += encoded_concat

        encoded_botleneck = cls.encode_convs(decoded_chromosome[1], real=real)

        if real:
            encoded_chromosome.extend(encoded_botleneck)
        else:
            encoded_chromosome += encoded_botleneck

        return encoded_chromosome

    # ========================
    # NOTE: Funciones de decodificación
    # ========================
    @staticmethod
    def decode_gene(value: Union[float, str], options: dict[str, Union[int, bool, str]], real: bool) -> Union[int, bool, str]:
        """
        Para real:
            Pasa el tamaño del diccionario a un rango de 0 a 1, selecciona el valor equivalente a value y lo devuelve
        Para binario:
            Devuelve el valor en la posición 'value' del diccionario

        Parameters
        ----------
        value : float or str
            Valor a decodificar
        options : dict
            Opciones de decodificación
        real : bool
            Indica si la codificación del gen es real o binaria

        Returns
        -------
        int or bool or str
            Valor encontrado en las opciones
        """
        if not real:
            if value in options:
                return options[value]
            else:
                return options[list(options.keys())[-1]]
        else:
            step = 1 / len(options)

            for i in range(len(options)):
                cota_inf = step * i
                cota_sup = step * (i + 1)

                if cota_inf <= value < cota_sup:
                    return list(options.values())[i]

            return list(options.values())[-1]

    @classmethod
    def decode_convs(cls, convs: Union[list[float], str], real: bool) -> list[tuple[int, int, str]]:
        """
        Decodifica una lista de convoluciones

        Parameters
        ----------
        conv : list or str
            Lista de convoluciones
        real : bool
            Indica si la codificación de las convoluciones es real o binaria

        Returns
        -------
        list
            Lista de convoluciones decodificadas
        """
        return [
            (
                Chromosome.decode_gene(  # f
                    value=convs[i] if real else convs[i:i + 4],
                    options=cls.FILTERS,
                    real=real
                ),
                Chromosome.decode_gene(  # s
                    value=convs[i + 1] if real else convs[i + 4:i + 6],
                    options=cls.KERNEL_SIZES,
                    real=real
                ),
                Chromosome.decode_gene(  # a
                    value=convs[i + 2] if real else convs[i + 6:i + 10],
                    options=cls.ACTIVATION_FUNCTIONS,
                    real=real
                )
            )
            for i in range(0, len(convs), 3 if real else 10)
        ]

    @classmethod
    def decode_layer(cls, layer: Union[list[float], str], real: bool) -> tuple[tuple[list[tuple[int, int, str]],
                                                                               str],
                                                                               tuple[list[tuple[int, int, str]],
                                                                               bool]]:
        """
        Decodifica una capa del cromosoma

        Parameters
        ----------
        layer : list or str
            Capa a decodificar
        real : bool
            Indica si la codificación de la capa es real o binaria

        Returns
        -------
        tuple
            Capa decodificada
        """
        encoder = layer[:len(layer) // 2]
        decoder = layer[len(layer) // 2:]

        pooling = Chromosome.decode_gene(
            value=encoder[-1],
            options=cls.POOLINGS,
            real=real
        )
        concat = Chromosome.decode_gene(
            value=decoder[-1],
            options=cls.CONCATENATION,
            real=real
        )

        # Decodificamos encoder
        decoded_convolutions = cls.decode_convs(
            convs=encoder[0:len(encoder) - 1],
            real=real
        )
        # Decodificamos decoder
        decoded_deconvolutions = cls.decode_convs(
            convs=decoder[0:len(decoder) - 1],
            real=real
        )

        return ((decoded_convolutions, pooling), (decoded_deconvolutions, concat))

    @classmethod
    def decode_chromosome(
        cls, chromosome: Union[list[float], str],
        layer_len: int,
        bottleneck_len: int,
        real: bool
    ) -> tuple[list[tuple[tuple[list[tuple[int, int, str]],
                                str],
                          tuple[list[tuple[int, int, str]],
                                bool]]],
               list[tuple[int, int, str]]]:
        """
        Transforma el cromosoma en una lista con los valores de las capas, entendible para el humano y siendo un paso previo a la creación del modelo

        Parameters
        ----------
        chromosome : list or str
            Cromosoma a decodificar
        layer_len : int
            Longitud de una capa
        bottleneck_len : int
            Longitud del cuello de botella
        real : bool
            Indica si la codificación del cromosoma es real o binaria

        Returns
        -------
        tuple
            Cromosoma decodificado
        """
        decoded_layers = [
            cls.decode_layer(
                layer=chromosome[i:i + layer_len],
                real=real
            )
            for i in range(0, len(chromosome) - bottleneck_len, layer_len)
        ]

        decoded_bottleneck = cls.decode_convs(
            convs=chromosome[len(chromosome) - bottleneck_len:len(chromosome)],
            real=real
        )

        return (decoded_layers, decoded_bottleneck)

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

        self.__decoded = self.decode_chromosome(
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

        self.__real: list[float] = self.encode_chromosome(
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
            self.__binary = self.unzip_binary(binary)
            # Limpiamos las caches caducadas
            self.data_loader = "0"
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

        self.__binary: str = self.encode_chromosome(
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
        self.aptitude = None
        self.__unet = None

        # Si el cromosoma no está definido, lo generamos
        if not self.__decoded:
            self.set_decoded()

        self.__unet = UNet(
            decoded_chromosome=self.__decoded,
            **kwargs
        )

    def set_aptitude(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, metric: str = "iou", **kwargs: float):
        """
        Evalúa el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con las imágenes a evaluar, by default None
        metric : str, optional
            Métrica a utilizar para calcular la pérdida, by default "iou"

            Opciones:
                - "iou"
                - "dice"
                - "dice crossentropy"
                - "accuracy"
        **kwargs : float
            Argumentos adicionales para las funciones de pérdida:
            - smooth : (float) Valor para suavizar la división
            - threshold : (float) Umbral para considerar si un píxel es 1 o 0

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

            data_loader = TorchDataLoader(self.data_loader)
        elif isinstance(data_loader, str):
            data_loader = TorchDataLoader(data_loader)

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
            return self.zip_binary(self.__binary)

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

    def get_aptitude(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, metric: str = "iou", **kwargs: float) -> float:
        """
        Devuelve la aptitud del cromosoma

        Parameters
        ----------
        data_loader : TorchDataLoader or str, optional
            DataLoader con las imágenes a evaluar, by default None
        metric : str, optional
            Métrica a utilizar para calcular la pérdida, by default "iou"
        **kwargs : float
            Argumentos adicionales para las funciones de pérdida:
            - smooth : (float) Valor para suavizar la división
            - threshold : (float) Umbral para considerar si un píxel es 1 o 0

        Returns
        -------
        float
            Aptitud del cromosoma
        """
        if self.aptitude is None or data_loader:
            self.set_aptitude(data_loader, metric, **kwargs)

        return self.aptitude

    # ========================
    # NOTE: Funciones de la UNet
    # ========================
    def train_unet(self, data_loader: Union[TorchDataLoader, str], **kwargs: Union[str, float, int, bool]):
        """
        Entrena el modelo UNet

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con los datos de entrenamiento y validación
        **kwargs : str or int or float or bool
            Argumentos adicionales para el entrenamiento:
            - metric : (str) Métrica a utilizar para calcular la pérdida. ("iou", "dice", "dice crossentropy" o "accuracy")
            - lr : (float) Tasa de aprendizaje
            - epochs : (int) Número de épocas
            - show_val : (bool) Si mostrar los resultados de la validación en cada epoch
            - print_every : (int) Cada cuántos pasos se imprime el resultado
        """
        if not self.__unet:
            self.set_unet()

        if isinstance(data_loader, str):
            data_loader = TorchDataLoader(data_loader)

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

    def show_results(self, data_loader: Optional[Union[TorchDataLoader, str]] = None, save: bool = False, name: Optional[str] = None):
        """
        Muestra los resultados del modelo UNet actual

        Parameters
        ----------
        data_loader : TorchDataLoader or str
            DataLoader con las imágenes a evaluar
        """
        if not self.__unet:
            self.set_unet()

        if not data_loader:
            if not self.data_loader:
                raise ValueError("No se ha especificado un DataLoader")

            data_loader = TorchDataLoader(self.data_loader)
        elif isinstance(data_loader, str):
            data_loader = TorchDataLoader(data_loader)

        if save and not name:
            name = self.__str__() + ".png"

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

        save_model(
            model=self.__unet,
            name=name
        )
        print(f"Modelo guardado como {name}")


if __name__ == "__main__":
    print("=== Prueba de codificación y decodificación ===")
    unet_paper = (
        [  # layers: [convs+deconvs, convs+deconvs, ...]
            (  # convs+deconvs: [nconvs+pooling, nconvs+concat]
                (  # nconvs+pooling: [nconvs, pooling]
                    [  # nconvs: [conv, conv, ...]
                        (64, 3, "relu"),  # conv: [f, s, a]
                        (64, 3, "relu")
                    ],
                    # pooling
                    "max"
                ),
                (  # nconvs+concat: [nconvs, concat]
                    [  # nconvs: [conv, conv, ...]
                        (64, 3, "relu"),  # conv: [f, s, a]
                        (64, 3, "relu")
                    ],
                    # concat
                    True
                )
            ),
            (
                (
                    [
                        (128, 3, "relu"),
                        (128, 3, "relu")
                    ],
                    "max"
                ),
                (
                    [
                        (128, 3, "relu"),
                        (128, 3, "relu")
                    ],
                    True
                )
            ),
            (
                (
                    [
                        (256, 3, "relu"),
                        (256, 3, "relu")
                    ],
                    "max"
                ),
                (
                    [
                        (256, 3, "relu"),
                        (256, 3, "relu")
                    ],
                    True
                )
            ),
            (
                (
                    [
                        (512, 3, "relu"),
                        (512, 3, "relu")
                    ],
                    "max"
                ),
                (
                    [
                        (512, 3, "relu"),
                        (512, 3, "relu")
                    ],
                    True
                )
            )
        ],
        [  # bottleneck: [conv, conv, ...]
            (1024, 3, "relu"),  # conv: [f, s, a]
            (1024, 3, "relu")
        ]
    )

    c = Chromosome(
        max_layers=4,
        max_conv_per_layer=2,
        chromosome=unet_paper
    )

    print("Name:\n", c)
    print("Decoded:\n", c.get_decoded())
    print("Real:\n", c.get_real())
    print("Binary:\n", c.get_binary())

    # Comprobamos que la decodificación y codificación sean correctas
    assert c.get_decoded() == unet_paper
    # Aunque tenga letras se trata de una codificación binaria
    # Solo está comprimida para que no sea tan larga
    assert (
        c.get_binary(zip=True) == "ARARAIQIQ4IMIGEGEHKDKBVBVB6Q6QPIPIPEHEA:188"
    )

    print("=== Prueba de entrenamiento y evaluación ===")
    # Distintas redes a elegir
    unet_paper = "ARARAIQIQ4IMIGEGEHKDKBVBVB6Q6QPIPIPEHEA:188"
    unet_paper_mini = "AJAJAEQEQWIGIDEDEF2B2A5A5BKQKQFIFIKECEA:188"
    unet_rara = "AE6HZLHCTEYFIMM24G3TPQWZ4AS5CUI:146"
    c = Chromosome(
        max_layers=4,
        max_conv_per_layer=2,
        chromosome=unet_rara
    )

    # Mostramos la arquitectura a generar
    print(c.get_decoded())

    # Al entrenarla se generará el modelo UNet automáticamente
    c.train_unet(
        data_loader="road",
        epochs=5
    )

    # Si no especificamos un DataLoader, se usará el dataloader con el que se entrenó
    c.show_results()
    c.show_results("carvana")

    # Tampoco es necesario especificar el DataLoader si ya se ha entrenado
    print(c.get_aptitude())
    print(c.get_aptitude("carvana"))

    # Guardamos los resultados
    c.show_results(
        save=True,
        name="para el road.png"
    )
    c.show_results(
        "carvana",
        save=True,
        name="para carvana.png"
    )

    c.save_unet()
