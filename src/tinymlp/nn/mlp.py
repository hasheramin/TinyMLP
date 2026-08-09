from tinymlp.nn.layer import Layer


class MLP:
    """
    Multi-Layer Perceptron (MLP) consisting of a sequence of Layers.
    """

    def __init__(self, nin: int, nouts: list, activations: list = None):
        sz = [nin] + nouts
        self.layers = []

        if activations is None:
            # Default: relu for hidden layers, linear/relu for final output
            activations = ["relu"] * (len(nouts) - 1) + ["linear"]
        elif isinstance(activations, str):
            activations = [activations] * len(nouts)

        assert len(activations) == len(nouts), "Number of activations must match number of layers"

        for i in range(len(nouts)):
            act = activations[i]
            nonlin = act != "linear"
            self.layers.append(
                Layer(sz[i], sz[i + 1], nonlin=nonlin, activation=act)
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        layers_str = ", ".join(str(layer) for layer in self.layers)
        return f"MLP([\n  {layers_str}\n])"
