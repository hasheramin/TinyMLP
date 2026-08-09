from tinymlp.nn.neuron import Neuron


class Layer:
    """
    A layer of neurons.
    """

    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)})"
