# TinyMLP Documentation

Welcome to the **TinyMLP** technical documentation. This guide details the internal mechanics of the scalar automatic differentiation engine and neural network abstractions.

---

## Architecture Overview

TinyMLP implements **Reverse-Mode Automatic Differentiation** (autograd) on scalar values. 

The core data structures and workflows are organized as follows:

```
[Value]  ---> Wraps float, tracks children, op type, and _backward lambda
   |
[Neuron] ---> Vector of Value weights + Value bias + activation
   |
[Layer]  ---> Array of Neurons
   |
[MLP]    ---> Chain of Layers
```

## Quick Import

All public components can be imported directly from `tinymlp`:
```python
from tinymlp import Value, MLP, Layer, Neuron, SGD, mse_loss, binary_cross_entropy, relu, sigmoid, tanh
```

---

## Core Module (`tinymlp.core`)


### `Value(data, _children=(), _op="", label="")`

The fundamental object tracking scalar numbers and computation graphs.

#### Attributes:
- `data`: `float` - The numerical value stored in the node.
- `grad`: `float` - The computed derivative of the output scalar w.r.t this node. Default `0.0`.
- `_prev`: `set` - Direct parent nodes in the DAG.
- `_op`: `str` - Operation string identifier (e.g. `+`, `*`, `exp`, `tanh`).

#### Supported Operations:
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Functions: `exp()`, `log()`, `relu()`, `sigmoid()`, `tanh()`
- Differentiation: `backward()` - Triggers topological sort and runs the chain rule backwards through all nodes.

---

## Neural Network Module (`tinymlp.nn`)

### `Neuron(nin, nonlin=True, activation='relu')`
Calculates: $y = \text{act}(\sum_{i} w_i x_i + b)$.

### `Layer(nin, nout, **kwargs)`
Creates `nout` neurons, each taking `nin` inputs.

### `MLP(nin, nouts, activations=None)`
Constructs a sequential multi-layer perceptron.
- `parameters()`: Returns list of all trainable `Value` objects across layers.
- `zero_grad()`: Sets `grad = 0.0` for all trainable parameters.

---

## Loss Functions (`tinymlp.losses`)

- `mse_loss(y_pred, y_true)`: Computes Mean Squared Error $\frac{1}{N} \sum (y_{pred} - y_{true})^2$.
- `binary_cross_entropy(y_pred, y_true)`: Computes Binary Cross Entropy $-\frac{1}{N} \sum [y \log(p) + (1-y) \log(1-p)]$.

---

## Optimizer (`tinymlp.optim`)

### `SGD(params, lr=0.01, momentum=0.0)`
- `step()`: Updates weights using $p \leftarrow p - \eta \cdot \nabla p$.
- `zero_grad()`: Clears parameter gradients.

---

## Testing Guidelines

Run tests using pytest:
```bash
pytest tests/
```
Or run individual example scripts:
```bash
python examples/xor.py
python examples/regression.py
```
