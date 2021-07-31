# import sys 
# sys.path.append('../')
# sys.path.append('../../')

from optlssvm.orlssvm import OptimallyRegularizedLSSVM
from optlssvm.kernels import RBF
import numpy as np

class OptimallyRegularizedRBFLSSVM(OptimallyRegularizedLSSVM):

    def __init__(self,  rbf_width_values = None ):
        super().__init__(RBF()) # kernel must be RBF
        self._multiply_initial_rbf_width = False
        if rbf_width_values is None:
            self._multiply_initial_rbf_wdith = True
            self._rbf_width_values = np.logspace(-3,2,50)
        else:
            self._rbf_width_values = rbf_width_values

    @property
    def rbf_width_values(self):
        return self._rbf_width_values

    @rbf_width_values.setter
    def rbf_width_values(self, vals):
        self._rbf_width_values = vals

    # big chunk of code is from the parent class, copied here just to make
    # the RBF calculation more efficient
    def _optimise(self, x: np.array, y: np.array)-> None:
        self.x = x
        self.y = y
        dist = self.kernel.squared_distance(x,x)

        if self._multiply_initial_rbf_width:
            self.kernel.set_initial_width(self.x)
            sigma = self._rbf_width_values * self.kernel.width
        else:
            sigma = self._rbf_width_values
        mu_vals = self.mu_values
        muX = np.zeros(len(sigma))
        pressX = np.zeros(len(sigma))
        for k in range(len(sigma)):
            eigVal, V = np.linalg.eigh(np.exp(-dist / sigma[k]**2).T)
            Vt_y = V.T.dot(y)
            Vt_sqr = V.T ** 2
            xi = (V.sum(axis=0)).T
            xi2 = xi ** 2
            PRESS = np.zeros(len(mu_vals))
            for i in range(len(mu_vals)):
                denom = eigVal + mu_vals[i]
                u = xi/denom
                g = eigVal/denom
                sm = -(xi2 /denom).sum()
                theta = Vt_y/denom + (u.dot(Vt_y)/sm)*u
                h = Vt_sqr.T.dot(g) + (V.dot(u*eigVal)-1)*V.dot(u)/sm
                f = V.dot(eigVal*theta) - sum(u*Vt_y)/ sm
                loo_resid = (y - f) / (1 - h)
                PRESS[i] = (loo_resid ** 2).sum()
            muX[k] = mu_vals[PRESS.argmin()]
            pressX[k] = min(PRESS)
            print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f" %
                    (sigma[k], muX[k], pressX[k]))
        muOpt = muX[pressX.argmin()]
        sigOpt = sigma[pressX.argmin()]
        print("\nOptimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" %
                (sigOpt, muOpt))
        return sigOpt, muOpt

    def fit(self, x: np.array, y: np.array)-> None:
        self.x = x
        self.y = y
        self.kernel.width, self.mu = self._optimise(x,y)
        print("training with opt parameters...")
        ## using the grand parent, i.e. LSSVM
        super(OptimallyRegularizedLSSVM, self).fit(self.x, self.y)




