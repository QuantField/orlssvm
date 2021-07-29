from .kernel import Kernel

class Polynomial(Kernel):
    def __init__(self, order, ofset):
        super().__init__('Polynomial')
        self.__order = order
        self.__ofset = ofset

    def evaluate(self, x1, x2):
        return (x2.dot(x1.T) + self.__ofset) ** self.__order

    def params(self):
        s = "Order = " + str(self.__order) + "   Offset = " + str(self.__ofset)
        return s

    def __repr__(self):
        return ("Polynomial(order= {}, offset= {})".format( self.__order,
            self.__ofset))