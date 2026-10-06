import math

from engine import Value
from gradcheck import gradcheck


def test_tanh():
    assert Value(0.5).tanh().data == math.tanh(0.5)
    assert gradcheck(lambda v: v[0].tanh(), [Value(0.5)]) < 1e-6
    assert gradcheck(lambda v: v[0].tanh(), [Value(-2.0)]) < 1e-6


def test_relu():
    assert Value(2.0).relu().data == 2.0
    assert Value(-2.0).relu().data == 0.0
    # check away from the kink at 0, where the slope jumps from 0 to 1
    assert gradcheck(lambda v: v[0].relu(), [Value(2.0)]) < 1e-6
    assert gradcheck(lambda v: v[0].relu() * 3, [Value(-2.0)]) == 0.0


def test_exp():
    assert Value(1.0).exp().data == math.e
    assert gradcheck(lambda v: v[0].exp(), [Value(0.7)]) < 1e-6


def test_log():
    assert Value(math.e).log().data == 1.0
    assert gradcheck(lambda v: v[0].log(), [Value(2.5)]) < 1e-6


def test_mixed_expression():
    inputs = [Value(0.5), Value(-0.8), Value(0.3)]
    assert gradcheck(lambda v: ((v[0] * v[1] + v[2]) ** 2).tanh(), inputs) < 1e-6


def test_everything_together():
    inputs = [Value(1.2), Value(0.4)]

    def f(v):
        a, b = v
        return (a.exp() / (1 + b ** 2)).log() - (a - b).relu() + (2 * b).tanh()

    assert gradcheck(f, inputs) < 1e-6
