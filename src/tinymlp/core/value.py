import math


class Value:
    """
    Stores a scalar value and its gradient for reverse-mode automatic differentiation.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        if self.label:
            return f"Value(label={self.label}, data={self.data}, grad={self.grad})"
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return Value(other) * (self**-1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        x = self.data
        assert x > 0, "logarithm only defined for positive values"
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Computes gradients for all nodes in the computational graph via reverse-mode AD.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one node at a time and apply chain rule to calculate gradients
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
