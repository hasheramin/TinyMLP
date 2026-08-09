import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.nn.layer import Layer
from tinymlp.core.value import Value



def test_layer_forward():
    layer = Layer(3, 2, nonlin=False)
    assert len(layer.neurons) == 2
    assert len(layer.parameters()) == 8  # 2 neurons * (3 weights + 1 bias)

    x = [1.0, 2.0, 3.0]
    out = layer(x)
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], Value)
    assert isinstance(out[1], Value)


def test_layer_single_output():
    layer = Layer(2, 1, nonlin=False)
    x = [1.0, 2.0]
    out = layer(x)
    assert isinstance(out, Value)
