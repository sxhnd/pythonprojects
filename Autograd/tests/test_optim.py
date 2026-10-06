from engine import Value
from optim import SGD


def minimize_square(optimizer_class, steps, **kwargs):
    # Minimize x**2 starting from x = 5. The minimum is at x = 0.
    x = Value(5.0)
    optimizer = optimizer_class([x], **kwargs)
    for _ in range(steps):
        loss = x ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return x.data


def test_sgd_single_step():
    # grad of x**2 at 5 is 10, so one step moves x to 5 - 0.1 * 10
    x = Value(5.0)
    optimizer = SGD([x], lr=0.1)
    (x ** 2).backward()
    optimizer.step()
    assert abs(x.data - 4.0) < 1e-12


def test_sgd_minimizes_square():
    assert abs(minimize_square(SGD, steps=100, lr=0.1)) < 1e-6


def test_sgd_zero_grad():
    x = Value(5.0)
    optimizer = SGD([x])
    (x ** 2).backward()
    optimizer.zero_grad()
    assert x.grad == 0.0
