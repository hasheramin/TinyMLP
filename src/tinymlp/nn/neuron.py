import random
from tinymlp.core.value import Value


class Neuron:
    """
    A single artificial neuron with weights, bias, and optional activation.
    """

    def __init__(self, nin: int, nonlin: bool = True, activation: str = "relu"):
        # Initialize weights randomly in [-1, 1] scaled by sqrt(nin) for basic Xavier-like init
        scale = 1.0 / (nin**0.5) if nin > 0 else 1.0
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
        self.activation = activation.lower() if isinstance(activation, str) else activation

    def __call__(self, x):
        # x is a list of inputs (floats or Values)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if not self.nonlin:
            return act

        if self.activation == "relu":
            return act.relu()
        elif self.activation == "sigmoid":
            return act.sigmoid()
        elif self.activation == "tanh":
            return act.tanh()
        elif callable(self.activation):
            return self.activation(act)
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        act_str = f", act='{self.activation}'" if self.nonlin else ", linear"
        return f"Neuron(nin={len(self.w)}{act_str})"
