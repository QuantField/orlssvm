from .kernel import Kernel

class Linear(Kernel):

    def __init__(self):
        super().__init__(self, 'Linear')

    def evaluate(self, x1, x2):
        return x2.dot(x1.T)

    def __repr__(self):
        return "Linear()"
        