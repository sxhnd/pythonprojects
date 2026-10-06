from engine import Value


def relative_error(a, b):
    # Relative, not absolute: a gap of 0.001 is huge for a gradient of 0.002
    # and negligible for a gradient of 1000.
    return abs(a - b) / max(abs(a), abs(b), 1e-8)


def numerical_gradient(f, inputs, i, eps=1e-6):
    # Central difference: its error shrinks with eps**2, while the one-sided
    # (f(w + eps) - f(w)) / eps only shrinks with eps.
    original = inputs[i].data
    inputs[i].data = original + eps
    plus = f(inputs).data
    inputs[i].data = original - eps
    minus = f(inputs).data
    inputs[i].data = original
    return (plus - minus) / (2 * eps)


def gradcheck(f, inputs, eps=1e-6):
    """Return the largest relative error between autograd and numerical gradients.

    f takes the list of input Values and returns a single output Value.
    Each input costs two extra forward passes, so this is for tests only.
    """
    for x in inputs:
        x.grad = 0.0
    f(inputs).backward()

    errors = []
    for i, x in enumerate(inputs):
        numeric = numerical_gradient(f, inputs, i, eps)
        errors.append(relative_error(x.grad, numeric))
    return max(errors)
