"""
Paquete con clases útiles para la representación en cromosomas de las
arquitecturas de redes neuronales

Clases
------
- Chromosome: Clase que representa un cromosoma de la población de un algoritmo genético

Funciones
---------
- layer_is_identity: Determina si una capa es una capa de identidad
- conv_is_identity: Determina si una convolución es una convolución de identidad

Constantes
----------
- MAX_LAYERS: Número máximo de capas que puede tener un cromosoma
- MAX_CONVS_PER_LAYER: Número máximo de convoluciones que puede tener una capa
"""
from .chromosome import Chromosome
from .functions import layer_is_identity, conv_is_identity
from .constants import MAX_LAYERS, MAX_CONVS_PER_LAYER

__all__ = [
    "Chromosome",
    "layer_is_identity", "conv_is_identity",
    "MAX_LAYERS", "MAX_CONVS_PER_LAYER"
]


# # ESTRUCTURA DE UN CROMOSOMA DECODIFICADO (solo para referencia)
# from typing import Optional

# tipo: tuple[list[Optional[tuple[tuple[list[Optional[tuple[int, int, str]]],
#                                       str],
#                                 tuple[list[Optional[tuple[int, int, str]]],
#                                       bool]]]],
#             list[Optional[tuple[int, int, str]]]]
# decoded = (
#     # chromosome: [layers, bottleneck]
#     [  # layers: [convs+deconvs, convs+deconvs, ...]
#         None,  # Identity layer (always before the actual layers)
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
#                     None,  # Identity conv (always before the actual convs)
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
