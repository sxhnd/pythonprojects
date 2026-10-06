from engine import Value
from optim import SGD, Adam, Momentum


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


def test_momentum_first_step_matches_sgd():
    # velocity starts at 0, so the first step is the same as plain SGD
    x = Value(5.0)
    optimizer = Momentum([x], lr=0.1, beta=0.9)
    (x ** 2).backward()
    optimizer.step()
    assert abs(x.data - 4.0) < 1e-12


def test_momentum_minimizes_square():
    assert abs(minimize_square(Momentum, steps=300, lr=0.1, beta=0.9)) < 1e-4


def test_momentum_beats_sgd_with_a_small_learning_rate():
    # With a small lr, SGD crawls. Momentum builds up speed in the
    # direction the gradient keeps pointing.
    sgd_x = minimize_square(SGD, steps=100, lr=0.01)
    momentum_x = minimize_square(Momentum, steps=100, lr=0.01, beta=0.9)
    assert abs(momentum_x) < abs(sgd_x) / 10


def test_adam_minimizes_square():
    assert abs(minimize_square(Adam, steps=500, lr=0.1)) < 1e-4


def test_adam_first_step_is_about_lr_for_any_gradient_size():
    # With bias correction, m_hat = g and v_hat = g**2 after one step, so the
    # update is lr * g / |g| = lr, whether the gradient is tiny or huge.
    for scale in [1e-3, 1.0, 1e3]:
        x = Value(5.0)
        optimizer = Adam([x], lr=0.1)
        (scale * x ** 2).backward()
        optimizer.step()
        assert abs((5.0 - x.data) - 0.1) < 1e-6

