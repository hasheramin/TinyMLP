import math
from tinymlp.core.value import Value


def tanh(x):
    """
    Applies Hyperbolic Tangent (Tanh) activation function to input x (Value or float/int).
    """
    if isinstance(x, Value):
        return x.tanh()
    return math.tanh(float(x))
