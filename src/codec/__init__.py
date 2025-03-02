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
from .chromosome import Chromosome
from .encode import zip_binary, encode_chromosome, encode_convs, encode_gene
from .decode import unzip_binary, decode_chromosome, decode_layer, decode_convs, decode_gene
from .constants import FILTERS, KERNEL_SIZES, ACTIVATION_FUNCTIONS, POOLINGS, CONCATENATION

__all__ = [
    "Chromosome",
    "zip_binary", "encode_chromosome", "encode_convs", "encode_gene",
    "unzip_binary", "decode_chromosome", "decode_layer", "decode_convs", "decode_gene",
    "FILTERS", "KERNEL_SIZES", "ACTIVATION_FUNCTIONS", "POOLINGS", "CONCATENATION"
]

# # ESTRUCTURA DE UN CROMOSOMA DECODIFICADO (solo para referencia)
# tipo: tuple[list[tuple[tuple[list[tuple[int, int, str]],
#                              str],
#                        tuple[list[tuple[int, int, str]],
#                              bool]]],
#             list[tuple[int, int, str]]]
# decoded = (
#     # chromosome: [layers, bottleneck]
#     [  # layers: [convs+deconvs, convs+deconvs, ...]
#         (  # convs+deconvs: [nconvs+pooling, nconvs+concat]
#             (  # nconvs+pooling: [nconvs, pooling]
#                 [  # nconvs: [conv, conv, ...]
#                     (3, 2, "a"),  # conv: [f, s, a]
#                     (3, 2, "a")
#                 ],
#                 # pooling
#                 "p"
#             ),
#             (  # nconvs+concat: [nconvs, concat]
#                 [  # nconvs: [conv, conv, ...]
#                     (3, 2, "a"),  # conv: [f, s, a]
#                     (3, 2, "a")
#                 ],
#                 # concat
#                 True
#             )
#         ),
#         (
#             (
#                 [
#                     (3, 2, "a"),
#                     (3, 2, "a")
#                 ],
#                 "p"
#             ),
#             (
#                 [
#                     (3, 2, "a"),
#                     (3, 2, "a")
#                 ],
#                 False
#             )
#         ),
#     ],
#     [  # bottleneck: [conv, conv, ...]
#         (3, 2, "a"),  # conv: [f, s, a]
#         (3, 2, "a")
#     ]
# )
