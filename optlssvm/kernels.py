
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np 


class EvaluateInterface(ABC):
    @abstractmethod
    def evaluate(self, x1: np.array, x2:np.array) -> np.array:
        pass


@dataclass 
class Linear(EvaluateInterface):
    def evaluate(self, x1: np.array, x2:np.array) -> np.array:
        return x2.dot(x1.T)


@dataclass
class Polynomial(EvaluateInterface):
    order: float 
    ofset: float 

    def evaluate(self, x1: np.array, x2:np.array) -> np.array:
        return (x2.dot(x1.T) + self.ofset) ** self.order


@dataclass
class RBF(EvaluateInterface):
    width:float = 0.5
    
    # Good starting point for the width
    def set_initial_width(self, trData: np.array) -> None:
        """Setting initial RBF width.

        The default value is the Euclidean norm of the standard
        deviations of the training variables, times 0.5
        """
        self.width = 0.5 * np.linalg.norm(trData.std(0))

    def squared_distance(self, x1: np.array, x2: np.array) -> np.array:
        """Efficient way of calculating (x1 -x2) squared.

        :param x1: n1 rows and m columns
        :param x2: n2 rows and m columns
        :return: (x1-x2) squared element wise
        """
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        p = (x1 ** 2).sum(axis=1).reshape(-1,1)
        par1 = p.dot(np.ones([1, n2]))
        if (x1 is x2):
            par2 = par1.T
        else:
            q = (x2 ** 2).sum(axis=1)
            q.shape = (1, n2)
            par2 = np.ones([n1, 1]).dot(q)
        return (par1 + par2 - 2 * x1.dot(x2.T))

    def evaluate(self, x1: np.array, x2:np.array) -> np.array:
        """Evaluate the RBF dot product Kernel matrix

        :param x1: n1 rows and m columns
        :param x2: n2 rows and m columns
        :return: RBF Kernel matrix
        """
        w = (self.width) ** 2
        K = self.squared_distance(x1, x2)
        return np.exp(-K / w).T


# Factory class
class Kernels:
    @staticmethod
    def generate_kernel(kerntype: str) -> EvaluateInterface:
        kerntype = kerntype.strip().lower()
        if kerntype=='linear':
            return Linear
        elif kerntype=='polynomial':
            return Polynomial
        elif kerntype=='rbf':
            return RBF
        else:
            raise ValueError('Kernel name not in [linear, polynomial, rbf]')    


def main():
    rbf = Kernels.generate_kernel('rbf')
    kern = rbf(1.0)
    print(kern)
    lin = Kernels.generate_kernel('linear')
    kern = lin()
    print(kern)
    pol = Kernels.generate_kernel('polynomial')
    kern = pol(2, 0.5)
    print(kern)

if __name__=='__main__':
    main()