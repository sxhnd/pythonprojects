import math


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


class Adam:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(self.params)  # running average of the gradient
        self.v = [0.0] * len(self.params)  # running average of the squared gradient
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        self.t += 1  # t starts at 1 so 1 - beta**t is never 0
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            # m and v start at 0, so early on they are pulled toward 0.
            # Dividing by 1 - beta**t undoes that pull.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
