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


def test_value_autograd_simple():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    d = e + c
    f = Value(-2.0, label='f')
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
    b = a ** 3
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


def test_value_division_subtraction():
    a = Value(10.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 0.5
    assert b.grad == -2.5
