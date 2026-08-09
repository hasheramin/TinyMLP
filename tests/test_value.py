import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.core.value import Value


def test_value_basic_ops():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    assert d.data == 4.0


def test_value_all_arithmetic_operators():
    a = Value(4.0)
    b = Value(2.0)

    # Addition
    assert (a + b).data == 6.0
    assert (a + 3.0).data == 7.0
    assert (3.0 + a).data == 7.0

    # Subtraction
    assert (a - b).data == 2.0
    assert (a - 1.0).data == 3.0
    assert (10.0 - a).data == 6.0
    assert (-a).data == -4.0

    # Multiplication
    assert (a * b).data == 8.0
    assert (a * 2.5).data == 10.0
    assert (2.5 * a).data == 10.0

    # Division
    assert (a / b).data == 2.0
    assert (a / 2.0).data == 2.0
    assert (12.0 / a).data == 3.0

    # Power
    assert (a**3).data == 64.0


def test_value_autograd_simple():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    d = e + c
    f = Value(-2.0, label="f")
    L = d * f

    L.backward()

    assert L.data == -8.0
    assert f.grad == 4.0
    assert d.grad == -2.0
    assert c.grad == -2.0
    assert e.grad == -2.0
    assert a.grad == 6.0   # dL/da = f * b = -2 * -3 = 6
    assert b.grad == -4.0  # dL/db = f * a = -2 * 2 = -4


def test_value_pow():
    a = Value(3.0)
    b = a**3
    b.backward()
    assert b.data == 27.0
    assert a.grad == 27.0  # d(a^3)/da = 3*a^2 = 3*9 = 27


def test_value_exp_log():
    a = Value(2.0)
    b = a.exp()
    b.backward()
    assert abs(b.data - math.exp(2.0)) < 1e-6
    assert abs(a.grad - math.exp(2.0)) < 1e-6

    x = Value(5.0)
    y = x.log()
    y.backward()
    assert abs(y.data - math.log(5.0)) < 1e-6
    assert abs(x.grad - 0.2) < 1e-6


def test_value_sigmoid_numerical_stability():
    # Large positive input
    v_pos = Value(1000.0)
    s_pos = v_pos.sigmoid()
    assert s_pos.data == 1.0
    s_pos.backward()
    assert v_pos.grad == 0.0

    # Large negative input (should not overflow math.exp)
    v_neg = Value(-1000.0)
    s_neg = v_neg.sigmoid()
    assert s_neg.data == 0.0
    s_neg.backward()
    assert v_neg.grad == 0.0


def test_value_repeated_backward():
    x = Value(3.0)
    y = x**2
    y.backward()
    assert x.grad == 6.0

    # Gradient accumulation if not cleared
    y2 = x**2
    y2.backward()
    assert x.grad == 12.0

    # Clearing gradient
    x.grad = 0.0
    y3 = x**2
    y3.backward()
    assert x.grad == 6.0
