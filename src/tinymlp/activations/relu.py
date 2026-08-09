from tinymlp.core.value import Value


def relu(x):
    """
    Applies Rectified Linear Unit (ReLU) activation to input x (Value or float/int).
    """
    if isinstance(x, Value):
        return x.relu()
    return max(0.0, float(x))
