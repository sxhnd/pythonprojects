from engine import Value
from gradcheck import gradcheck, numerical_gradient, relative_error


def test_relative_error_scales_with_size():
    assert relative_error(1000.0, 1000.001) < 1e-6
    assert relative_error(0.002, 0.003) > 0.1


def test_numerical_gradient_restores_input():
    x = Value(3.0)
    numerical_gradient(lambda v: v[0] * v[0], [x], 0)
    assert x.data == 3.0


def test_add_passes():
    inputs = [Value(2.0), Value(-3.0)]
    assert gradcheck(lambda v: v[0] + v[1], inputs) < 1e-6


def test_mul_passes():
    inputs = [Value(2.0), Value(-3.0)]
    assert gradcheck(lambda v: v[0] * v[1], inputs) < 1e-6


def test_mixed_expression_passes():
    inputs = [Value(2.0), Value(-3.0), Value(10.0)]
    assert gradcheck(lambda v: v[0] * v[1] + v[2], inputs) < 1e-6


def test_reused_input_passes():
    inputs = [Value(1.5)]
    assert gradcheck(lambda v: v[0] * v[0] * v[0] + v[0], inputs) < 1e-6


def test_catches_a_wrong_gradient():
    def bad_square(x):
        out = Value(x.data ** 2, (x,))

        def _backward():
            x.grad += x.data * out.grad  # wrong: should be 2 * x.data
        out._backward = _backward
        return out

    inputs = [Value(3.0)]
    assert gradcheck(lambda v: bad_square(v[0]), inputs) > 1e-2
