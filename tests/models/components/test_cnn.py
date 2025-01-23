import pytest
import torch

from models.components.cnn import CNN, ConvLayerParams, FcLayerParams


@pytest.mark.parametrize(("input_channels", "input_size"), [(1, (28, 28)), (3, (30, 30))])
def test_cnn_input_shape(input_channels: int, input_size: tuple[int, int]) -> None:
    batch_size = 16
    dummy_input = torch.randn(batch_size, input_channels, *input_size)

    num_classes = 10
    cnn = CNN(input_channels=input_channels, input_size=input_size, num_classes=num_classes)
    output = cnn(dummy_input)

    assert output.shape == (batch_size, num_classes)


@pytest.mark.parametrize("num_classes", [10, 20])
def test_cnn_num_classes(num_classes: int) -> None:
    batch_size = 16
    input_channels = 1
    input_size = (28, 28)
    dummy_input = torch.randn(batch_size, input_channels, *input_size)

    cnn = CNN(input_channels=input_channels, input_size=input_size, num_classes=num_classes)
    output = cnn(dummy_input)

    assert output.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    ("conv_layers", "fc_layers"),
    [
        (
            [
                ConvLayerParams(out_channels=32, kernel_size=3, stride=1, padding="same"),
                ConvLayerParams(out_channels=64, kernel_size=3, stride=1, padding="same"),
            ],
            [FcLayerParams(out_features=64)],
        ),
        (
            [
                ConvLayerParams(out_channels=32, kernel_size=3, stride=1, padding="same"),
                ConvLayerParams(out_channels=64, kernel_size=5, stride=2, padding=1),
            ],
            [FcLayerParams(out_features=128), FcLayerParams(out_features=64)],
        ),
    ],
)
def test_cnn_layers(conv_layers: list[ConvLayerParams], fc_layers: list[FcLayerParams]) -> None:
    batch_size = 16
    input_channels = 1
    input_size = (28, 28)
    dummy_input = torch.randn(batch_size, input_channels, *input_size)

    num_classes = 10
    cnn = CNN(
        input_channels=input_channels,
        input_size=input_size,
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        num_classes=num_classes,
    )
    output = cnn(dummy_input)

    assert output.shape == (batch_size, num_classes)
