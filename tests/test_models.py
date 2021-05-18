import pytest
import torch
from models import CNNMnist


@pytest.fixture
def cnn_mnist():
    return CNNMnist()


def test_cnn_mnist(cnn_mnist):
    sim = torch.randn(10, 1, 28, 28)
    out = cnn_mnist(sim)
    assert out.shape == (10, 10)


