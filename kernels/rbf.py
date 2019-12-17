from .kernel import Kernel
import numpy as np

class RBF(Kernel):

    def __init__(self, sigma=0.5):
        super(RBF, self).__init__('RBF')
        self.__width = sigma
        self.__type = 'RBF'

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width

    # Good starting point for the width
    def set_initial_width(self, trData):
        """Setting initial RBF width.

        The default value is the Euclidean norm of the standard
        deviations of the training variables, times 0.5
        """
        self.__width = 0.5 * np.linalg.norm(trData.std(0))

    def squared_distance(self, x1, x2):
        """Efficient way of calculating (x1 -x2) squared.

        :param x1: n1 rows and m columns
        :param x2: n2 rows and m columns
        :return: (x1-x2) squared element wise
        """
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        p = (x1 ** 2).sum(axis=1)
        p.shape = (n1, 1)
        par1 = p.dot(np.ones([1, n2]))
        if (x1 is x2):
            par2 = par1.T
        else:
            q = (x2 ** 2).sum(axis=1)
            q.shape = (1, n2)
            par2 = np.ones([n1, 1]).dot(q)
        return (par1 + par2 - 2 * x1.dot(x2.T))

    def evaluate(self, x1, x2):
        """Evaluate the RBF dot product Kernel matrix

        :param x1: n1 rows and m columns
        :param x2: n2 rows and m columns
        :return: RBF Kernel matrix
        """
        w = (self.__width) ** 2
        K = self.squared_distance(x1, x2)
        return np.exp(-K / w).T

    def params(self):
        return 'RBF Width = ' + str(self.__width)[:6]