
from abc import ABC

class Kernel(ABC):

    def __init__(self, kerntype):
        self._type = "Not defined"
        s = kerntype.strip().upper()
        if s in ('LINEAR', 'POLYNOMIAL', 'RBF'):
            self._type = s
        else:
            print("invalid Kernel, only  LINEAR,POLYNOMIAL,RBF are allowed")

    @abc.abstractmethod
    def params(self):
        return self._type

    def __str__(self):
        return self.params()

    @abc.abstractmethod
    def evaluate(self, x1, x2):
        return
