from .or_lssvm import or_lssvm
from kernels.rbf import RBF
import numpy as np

class opt_rbf_lssvm(or_lssvm):

    def __init__(self):
        super().__init__(RBF())


    def fit(self, x, y):
        self.x = x
        self.y = y
        kern = self.kernel
        kern.setInitWidh(self.x)
        sig = kern.getWidth()
        sigma = (10 ** np.arange(-3, 2.25, 0.25)) * sig
        muX = np.zeros(len(sigma))
        pressX = np.zeros(len(sigma))
        for i in range(len(sigma)):
            ls = or_lssvm(RBF(sigma[i]))
            muX[i], pressX[i] = ls._optimise(x, y, echo=False)
            print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f" %
                  (sigma[i], muX[i], pressX[i]))
        muOpt = muX[pressX.argmin()]
        sigOpt = sigma[pressX.argmin()]
        print("Optimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" %
              (sigOpt, muOpt))
        self.kernel, self.mu = RBF(sigOpt),  muOpt
        print("training with opt parameters...")
        super(or_lssvm, self).fit(self.x, self.y)




