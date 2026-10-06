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


def test_backward_sets_output_grad_to_one():
    a = Value(2.0)
    a.backward()
    assert a.grad == 1.0


def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_chain_rule_through_two_operations():
    # d = a*b + c, so dd/da = b, dd/db = a, dd/dc = 1
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    d.backward()
    assert d.data == 4.0
    assert a.grad == -3.0
    assert b.grad == 2.0
    assert c.grad == 1.0


def test_gradient_flows_through_a_deeper_chain():
    # e = (a*b) * c, so de/da = b*c
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    e = (a * b) * c
    e.backward()
    assert a.grad == 12.0
    assert b.grad == 8.0
    assert c.grad == 6.0
