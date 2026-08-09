"""
TinyMLP: A lightweight scalar automatic differentiation engine and neural network library.
"""

from tinymlp.core.value import Value
from tinymlp.nn.mlp import MLP
from tinymlp.nn.layer import Layer
from tinymlp.nn.neuron import Neuron

__version__ = "0.1.0"
__all__ = ["Value", "MLP", "Layer", "Neuron"]
