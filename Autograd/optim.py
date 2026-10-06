class SGD:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Momentum:
    # PyTorch form: v = beta * v + g, then w -= lr * v.
    # The classic form v = beta * v - lr * g, then w += v, gives the same
    # updates when lr is fixed; it only moves lr inside the velocity.
    def __init__(self, params, lr=0.1, beta=0.9):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [0.0] * len(self.params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.beta * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]
