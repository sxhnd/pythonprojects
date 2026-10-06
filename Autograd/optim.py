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
