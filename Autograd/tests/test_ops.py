from engine import Value
from gradcheck import gradcheck


def test_pow():
    x = Value(3.0)
    assert (x ** 2).data == 9.0
    assert gradcheck(lambda v: v[0] ** 3, [Value(1.5)]) < 1e-6
    assert gradcheck(lambda v: v[0] ** -2, [Value(1.5)]) < 1e-6
    assert gradcheck(lambda v: v[0] ** 0.5, [Value(2.0)]) < 1e-6


def test_neg():
    assert (-Value(3.0)).data == -3.0
    assert gradcheck(lambda v: -v[0], [Value(3.0)]) < 1e-6


def test_sub():
    assert (Value(5.0) - Value(3.0)).data == 2.0
    assert (Value(5.0) - 3).data == 2.0
    assert gradcheck(lambda v: v[0] - v[1], [Value(5.0), Value(3.0)]) < 1e-6


def test_div():
    assert (Value(6.0) / Value(3.0)).data == 2.0
    assert (Value(6.0) / 3).data == 2.0
    assert gradcheck(lambda v: v[0] / v[1], [Value(6.0), Value(-1.5)]) < 1e-6


def test_number_on_the_left():
    x = Value(4.0)
    assert (2 * x).data == 8.0
    assert (1 + x).data == 5.0
    assert (10 - x).data == 6.0
    assert (8 / x).data == 2.0
    assert gradcheck(lambda v: 2 * v[0] + 1 - 10 / v[0], [Value(4.0)]) < 1e-6


def test_sum_of_values():
    values = [Value(1.0), Value(2.0), Value(3.0)]
    total = sum(values)
    assert total.data == 6.0
    total.backward()
    assert [v.grad for v in values] == [1.0, 1.0, 1.0]
