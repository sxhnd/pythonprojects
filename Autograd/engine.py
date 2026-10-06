class Value:
    """A single number that remembers which Values it was computed from."""

    def __init__(self, data, _parents=()):
        self.data = data
        self.grad = 0.0
        self._parents = _parents

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other))

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
