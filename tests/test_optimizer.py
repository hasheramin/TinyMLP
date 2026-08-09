import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.core.value import Value
from tinymlp.optim.sgd import SGD



def test_sgd_optimizer():
    w = Value(5.0)
    optimizer = SGD([w], lr=0.1)

    # Loss = w^2 -> dLoss/dw = 2*w = 10
    loss = w ** 2
    optimizer.zero_grad()
    loss.backward()
    assert w.grad == 10.0

    optimizer.step()
    # w_new = 5.0 - 0.1 * 10.0 = 4.0
    assert abs(w.data - 4.0) < 1e-6

    optimizer.zero_grad()
    assert w.grad == 0.0
