from .or_lssvm import or_lssvm
from kernels.rbf import RBF
import numpy as np

class opt_rbf_lssvm(or_lssvm):

    def __init__(self, kern=RBF(), rbf_withs_values = None ):
        super().__init__(kern)
        self._multiply_initial_rbf_width = False
        if rbf_withs_values is None:
            self._multiply_initial_rbf_wdith = True
            self._rbf_width_values = np.logspace(-3,2,100)
        else:
            self._rbf_width_values = rbf_withs_values

    @property
    def rbf_width_values(self):
        return self._rbf_width_values

    @rbf_width_values.setter
    def rbf_width_values(self, vals):
        self._rbf_width_values = vals

    def fit(self, x, y):
        self.x = x
        self.y = y
        kern = self.kernel
        if self._multiply_initial_rbf_width:
            kern.set_initial_width(self.x)
            sigma = self._rbf_width_values * kern.width
        else:
            sigma = self._rbf_width_values
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




