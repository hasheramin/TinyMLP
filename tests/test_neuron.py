import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.nn.neuron import Neuron
from tinymlp.core.value import Value



def test_neuron_initialization():
    n = Neuron(3)
    assert len(n.w) == 3
    assert len(n.parameters()) == 4  # 3 weights + 1 bias


def test_neuron_forward():
    n = Neuron(2, nonlin=False)
    n.w[0].data = 2.0
    n.w[1].data = -1.0
    n.b.data = 0.5

    x = [1.0, 2.0]
    out = n(x)
    # 2.0*1.0 + (-1.0)*2.0 + 0.5 = 0.5
    assert isinstance(out, Value)
    assert out.data == 0.5
