from .lssvm import lssvm
import numpy as np

class or_lssvm(lssvm):

    muArray = np.logspace(-3, 2, 50)

    def _optimise(self, x, y, Mu=muArray, echo=True):
        self.x = x
        self.y = y
        eigVal, V = np.linalg.eigh(self.kernel.evaluate(x, x))
        Vt_y = V.T.dot(y)
        Vt_sqr = V.T ** 2
        xi = (V.sum(axis=0)).T
        xi2 = xi ** 2
        PRESS = np.zeros(len(Mu))
        for i in range(len(Mu)):
            u = xi / (eigVal + Mu[i])
            g = eigVal / (eigVal + Mu[i])
            sm = -(xi2 / (eigVal + Mu[i])).sum()
            theta = Vt_y / (eigVal + Mu[i]) + (u.dot(Vt_y) / sm) * u
            h = Vt_sqr.T.dot(g) + (V.dot(u * eigVal) - 1) * (V.dot(u)) / sm
            f = V.dot(eigVal * theta) - sum(u * Vt_y) / sm
            loo_resid = (y - f) / (1 - h)
            PRESS[i] = (loo_resid ** 2).sum()
            if echo:
                print("Mu= %2.4f  PRESS=%f"%(Mu[i],PRESS[i]))
        return Mu[PRESS.argmin()], min(PRESS)

    def fit(self, x, y, Mu=muArray):
        mu, press = self._optimise(x,y)
        self.mu = mu
        super().fit(x,y)



