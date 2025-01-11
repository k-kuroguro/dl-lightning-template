from typing import Literal, TypedDict

import torch
from torch import nn


class ConvLayerParams(TypedDict):
    out_channels: int
    kernel_size: int
    stride: int
    padding: int | Literal["same"]


class FcLayerParams(TypedDict):
    out_features: int


class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_size: tuple[int, int] = (28, 28),
        conv_layers: list[ConvLayerParams] = [
            ConvLayerParams(out_channels=32, kernel_size=3, stride=1, padding="same"),
            ConvLayerParams(out_channels=64, kernel_size=3, stride=1, padding="same"),
        ],
        dropout: float = 0.5,
        fc_layers: list[FcLayerParams] = [FcLayerParams(out_features=128)],
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_channels = input_channels
        current_size = list(input_size)

        for layer in conv_layers:
            padding = layer["padding"]
            if padding == "same":
                padding = layer["kernel_size"] // 2

            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout))

            current_channels = layer["out_channels"]
            for i in range(2):
                current_size[i] = (current_size[i] + 2 * padding - layer["kernel_size"]) // layer[
                    "stride"
                ] + 1
                current_size[i] = current_size[i] // 2

        flatten_size = current_channels * current_size[0] * current_size[1]
        current_features = flatten_size

        layers.append(nn.Flatten())
        for layer in fc_layers:
            layers.append(nn.Linear(current_features, layer["out_features"]))
            layers.append(nn.ReLU())
            current_features = layer["out_features"]

        layers.append(nn.Linear(current_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
