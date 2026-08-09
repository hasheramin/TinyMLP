import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.core.value import Value
from tinymlp.activations.relu import relu
from tinymlp.activations.sigmoid import sigmoid
from tinymlp.activations.tanh import tanh


def test_relu():
    assert relu(-2.0) == 0.0
    assert relu(3.5) == 3.5

    v1 = Value(-2.0)
    out1 = relu(v1)
    out1.backward()
    assert out1.data == 0.0
    assert v1.grad == 0.0

    v2 = Value(3.0)
    out2 = relu(v2)
    out2.backward()
    assert out2.data == 3.0
    assert v2.grad == 1.0


def test_sigmoid():
    v = Value(0.0)
    out = sigmoid(v)
    out.backward()
    assert abs(out.data - 0.5) < 1e-6
    assert abs(v.grad - 0.25) < 1e-6

    # Test numerical stability with extreme scalar inputs
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0


def test_tanh():
    v = Value(0.0)
    out = tanh(v)
    out.backward()
    assert abs(out.data - 0.0) < 1e-6
    assert abs(v.grad - 1.0) < 1e-6

    assert abs(tanh(0.0) - 0.0) < 1e-6
