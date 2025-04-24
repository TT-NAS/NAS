"""
Módulo con la clase UNet para la implementación de arquitecturas UNet dinámicas

Clases
------
- UNet: Implementación de la arquitectura UNet capaz de generarse a partir de un cromosoma decodificado
"""
from typing import Optional

import torch
from torch import nn, Tensor

from ..constants import CHANNELS


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
        from codec import layer_is_identity

        super().__init__()

        layers, bottleneck = decoded_chromosome
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        concats_sizes: list[int] = []

        for layer in layers:
            if layer_is_identity(layer, "decoded"):
                continue

            (convs, pooling), _ = layer
            convs_layer, in_channels = self.build_convs(convs, in_channels)
            self.encoder.append(convs_layer)
            concats_sizes.append(in_channels)

            if pooling == "average":
                self.encoder.append(
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
            else:
                self.encoder.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

        # Bottleneck
        self.bottleneck, in_channels = self.build_convs(
            bottleneck,
            in_channels
        )

        # Decoder
        self.concat_or_not = []

        for layer in layers[::-1]:
            if layer_is_identity(layer, "decoded"):
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
        from codec import conv_is_identity

        convs_layer = nn.Sequential()

        for i, conv in enumerate(convs):
            if conv_is_identity(conv, "decoded"):
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
