from .lssvm import lssvm
import numpy as np

class or_lssvm(lssvm):
    """Optimal Regularistion of LSSVM.

    The regularisation parameter mu is tuned as to minimise the PRESS statistic
    (proportional to the Leave oun out error).
    The range of values for searching the optimal mu is [10^-3,..,10^2], i.e
    np.logspace(-3, 2, 50)
    """

    def __init__(self, kern, mu_values = None):
        """or_lssvm constructor.

        :param kern: Kernel
        :param mu_values: range of regularisation parameters values
        """
        super().__init__(kern)
        if mu_values is None:
            self._mu_values = np.logspace(-3, 2, 20)
        else:
            self._mu_values = mu_values

    @property
    def mu_values(self):
        return self._mu_values

    @mu_values.setter
    def mu_values(self, mu_vals):
        self._mu_values = mu_vals

    def _optimise(self, x, y, echo=True):
        self.x = x
        self.y = y
        eigVal, V = np.linalg.eigh(self.kernel.evaluate(x, x))
        Vt_y = V.T.dot(y)
        Vt_sqr = V.T ** 2
        xi = (V.sum(axis=0)).T
        xi2 = xi ** 2
        mu_vals = self.mu_values
        PRESS = np.zeros(len(mu_vals))
        for i in range(len(mu_vals)):
            u = xi / (eigVal + mu_vals[i])
            g = eigVal / (eigVal + mu_vals[i])
            sm = -(xi2 / (eigVal + mu_vals[i])).sum()
            theta = Vt_y / (eigVal + mu_vals[i]) + (u.dot(Vt_y) / sm) * u
            h = Vt_sqr.T.dot(g) + (V.dot(u * eigVal) - 1) * (V.dot(u)) / sm
            f = V.dot(eigVal * theta) - sum(u * Vt_y) / sm
            loo_resid = (y - f) / (1 - h)
            PRESS[i] = (loo_resid ** 2).sum()
            if echo:
                print("Mu= %2.4f  PRESS=%f"%(mu_vals[i], PRESS[i]))
        return mu_vals[PRESS.argmin()], min(PRESS)

    def fit(self, x, y):
        mu, press = self._optimise(x,y)
        self.mu = mu
        super().fit(x,y)



