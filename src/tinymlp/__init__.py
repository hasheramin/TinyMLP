"""
TinyMLP: A lightweight scalar automatic differentiation engine and neural network library.
"""

from tinymlp.core.value import Value
from tinymlp.nn.neuron import Neuron
from tinymlp.nn.layer import Layer
from tinymlp.nn.mlp import MLP
from tinymlp.activations import relu, sigmoid, tanh
from tinymlp.losses import mse_loss, binary_cross_entropy
from tinymlp.optim import SGD

__version__ = "0.1.0"

__all__ = [
    "Value",
    "Neuron",
    "Layer",
    "MLP",
    "relu",
    "sigmoid",
    "tanh",
    "mse_loss",
    "binary_cross_entropy",
    "SGD",
]
