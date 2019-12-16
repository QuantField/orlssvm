
from abc import ABC, abstractmethod

class Kernel(ABC):

    def __init__(self, kerntype):
        self._type = "Not defined"
        s = kerntype.strip().upper()
        if s in ('LINEAR', 'POLYNOMIAL', 'RBF'):
            self._type = s
        else:
            print("invalid Kernel, only  LINEAR,POLYNOMIAL,RBF are allowed")

    @abstractmethod
    def params(self):
        return self._type

    def __str__(self):
        return self.params()

    @abstractmethod
    def evaluate(self, x1, x2):
       pass
