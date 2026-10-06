class Value:
    """A single number that remembers which Values it was computed from."""

    def __init__(self, data, _parents=()):
        self.data = data
        self.grad = 0.0
        self._parents = _parents
        self._backward = lambda: None  # leaf nodes have nothing to pass back

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            # d(out)/d(self) = 1 and d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            # d(out)/d(self) = other.data and d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only number exponents are supported"
        out = Value(self.data ** exponent, (self,))

        def _backward():
            # d(x**n)/dx = n * x**(n-1)
            self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    # Python calls these when the Value is on the right, as in 2 * x or 0 + x
    # (sum() starts from 0).
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        # Order the graph so every node comes after all the nodes it was built from.
        order = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    visit(parent)
                order.append(node)

        visit(self)

        # Walking that order in reverse means a node passes gradient back
        # only after it has received gradient from everything that uses it.
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
