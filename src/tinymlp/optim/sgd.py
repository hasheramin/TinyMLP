class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0] * len(self.params)

    def step(self):
        """
        Updates parameters based on gradients.
        """
        for i, p in enumerate(self.params):
            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
                p.data -= self.velocities[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Resets gradients of all parameters to zero.
        """
        for p in self.params:
            p.grad = 0.0
