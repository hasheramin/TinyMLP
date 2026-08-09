# TinyMLP

**TinyMLP** is a lightweight, pure Python scalar automatic differentiation (autograd) engine and neural network library built for education, experimentation, and deep learning fundamentals.

---

## Key Features

- **Scalar Autograd Engine**: Tracks computational graphs dynamically and computes gradients using reverse-mode automatic differentiation.
- **Neural Network Components**: Clean `Neuron`, `Layer`, and `MLP` abstractions similar to PyTorch.
- **Activation Functions**: `ReLU`, `Sigmoid`, and `Tanh`.
- **Loss Functions**: `MSE` (Mean Squared Error) and `CrossEntropy` (Binary Cross Entropy).
- **Optimizer**: `SGD` (Stochastic Gradient Descent).
- **Zero External Dependencies**: Standard library Python with optional `pytest` for testing.

---

## Directory Structure

```
TinyMLP/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── tinymlp/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── value.py
│       │   └── engine.py
│       ├── nn/
│       │   ├── __init__.py
│       │   ├── neuron.py
│       │   ├── layer.py
│       │   └── mlp.py
│       ├── activations/
│       │   ├── __init__.py
│       │   ├── relu.py
│       │   ├── sigmoid.py
│       │   └── tanh.py
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── mse.py
│       │   └── cross_entropy.py
│       └── optim/
│           ├── __init__.py
│           └── sgd.py
│
├── tests/
│   ├── __init__.py
│   ├── test_value.py
│   ├── test_neuron.py
│   ├── test_layer.py
│   ├── test_mlp.py
│   ├── test_activations.py
│   ├── test_losses.py
│   └── test_optimizer.py
│
├── examples/
│   ├── basic_mlp.py
│   ├── xor.py
│   └── regression.py
│
└── docs/
    └── README.md
```

---

## Quickstart

### 1. Installation

Install in editable mode:
```bash
pip install -e .
```

### 2. Autograd Example

```python
from tinymlp.core.value import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b
d = e + c
f = Value(-2.0, label='f')
L = d * f

L.backward()

print(f"L.data: {L.data}") # -8.0
print(f"a.grad: {a.grad}") # 6.0
print(f"b.grad: {b.grad}") # -4.0
```

### 3. Training an MLP on XOR

```python
from tinymlp.nn.mlp import MLP
from tinymlp.losses.mse import mse_loss
from tinymlp.optim.sgd import SGD

# 2 inputs -> hidden layer of 4 -> 1 output
model = MLP(2, [4, 1], activations=['relu', 'sigmoid'])
optimizer = SGD(model.parameters(), lr=0.5)

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [0.0, 1.0, 1.0, 0.0]

for epoch in range(100):
    y_pred = [model(x) for x in X]
    loss = mse_loss(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data:.4f}")
```

---

## Running Tests

Run all unit tests with pytest:

```bash
pytest tests/
```

---

## License

[MIT License](LICENSE)
