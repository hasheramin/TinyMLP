import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.core.value import Value
from tinymlp.losses.mse import mse_loss
from tinymlp.losses.cross_entropy import binary_cross_entropy



def test_mse_loss():
    y_pred = [Value(2.0), Value(3.0)]
    y_true = [1.0, 3.0]
    loss = mse_loss(y_pred, y_true)
    # loss = ((2-1)^2 + (3-3)^2)/2 = 0.5
    assert abs(loss.data - 0.5) < 1e-6

    loss.backward()
    assert abs(y_pred[0].grad - 1.0) < 1e-6  # d/d(p0) = (p0-t0) = 1.0
    assert abs(y_pred[1].grad - 0.0) < 1e-6


def test_bce_loss():
    y_pred = [Value(0.8), Value(0.2)]
    y_true = [1.0, 0.0]
    loss = binary_cross_entropy(y_pred, y_true)
    assert loss.data > 0.0
    loss.backward()
    assert y_pred[0].grad < 0.0  # increasing y_pred[0] lowers loss
    assert y_pred[1].grad > 0.0  # increasing y_pred[1] increases loss
