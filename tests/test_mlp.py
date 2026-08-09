import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.nn.mlp import MLP
from tinymlp.core.value import Value



def test_mlp_structure():
    model = MLP(3, [4, 4, 1], activations=['relu', 'relu', 'linear'])
    assert len(model.layers) == 3
    # Layer 0: 3->4 = 4*(3+1)=16 params
    # Layer 1: 4->4 = 4*(4+1)=20 params
    # Layer 2: 4->1 = 1*(4+1)=5 params
    # Total = 41 params
    assert len(model.parameters()) == 41


def test_mlp_forward_and_zero_grad():
    model = MLP(2, [3, 1], activations=['relu', 'linear'])
    x = [0.5, -1.0]
    out = model(x)
    assert isinstance(out, Value)

    out.backward()
    # verify gradients non-zero
    assert any(p.grad != 0.0 for p in model.parameters())

    model.zero_grad()
    # verify gradients cleared
    assert all(p.grad == 0.0 for p in model.parameters())
