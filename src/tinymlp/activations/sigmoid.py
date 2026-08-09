import math
from tinymlp.core.value import Value


def sigmoid(x):
    """
    Applies Sigmoid activation function to input x (Value or float/int).
    """
    if isinstance(x, Value):
        return x.sigmoid()
    val = float(x)
    if val >= 0:
        return 1.0 / (1.0 + math.exp(-val))
    z = math.exp(val)
    return z / (1.0 + z)

