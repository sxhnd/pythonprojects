from engine import Value


def test_add_and_mul_forward():
    assert (Value(2) * 3 + 1).data == 7


def test_add_two_values():
    assert (Value(2.0) + Value(5.0)).data == 7.0


def test_result_remembers_parents():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c._parents == (a, b)


def test_new_value_starts_with_zero_grad():
    assert Value(4.0).grad == 0.0
