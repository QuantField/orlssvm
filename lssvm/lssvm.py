import numpy as np
from  kernels.rbf import RBF
import copy

class lssvm:

    muArray = np.logspace(-3, 2, 50)

    def __init__(self, kern = RBF(), mu=0.1):
        self.alpha = None
        self.ntp = 0.0
        self.bias = 0.0
        self.x = None
        self.y = None
        self.mu = mu
        self.kernel = kern

    def copy(self):
        return copy.deepcopy(self)

    def fit(self, x, y):
        n = len(y)
        self.ntp = n
        self.x = x
        self.y = y
        K = self.kernel.evaluate(x, x)
        T = np.ones([n + 1, n + 1])
        T[n][n] = 0.0
        T[:n, :n] = K + self.mu * np.eye(n)
        Sol = np.linalg.solve(T, np.append(y, 0))
        self.alpha, self.bias = Sol[0:n], Sol[n]

    def predict(self, xt):
        P = xt.shape[0]
        N = 2000
        # scoring is performed in chunks, for obvious reasons
        chunks = int(P / N)
        pred = []
        for i in range(chunks + 1):
            start = i * N + (i != 0)
            stop = min(P - 1, (i + 1) * N)
            # print(" scoring chunk :",  start," - ", stop)
            chunk_index = list(range(start, stop + 1))
            K = self.kernel.evaluate(xt[chunk_index, :], self.x)
            prd = (K.T).dot(self.alpha) + self.bias
            pred = np.append(pred, prd)
        return pred

    def loo_residuals(self):
        """
        Caluculate Leave One Out residuals
        loo_resid = (y - yhat)/(1-diag(H))
        PRESS can be calculated as loo_resid.dot(loo_resid)
        :return:  loo_resid : Leave One Out residuals,
        """
        n = self.ntp
        K = self.kernel.evaluate(self.x, self.x)
        yhat = (K.T).dot(self.alpha) + self.bias
        T = np.ones([n + 1, n + 1])
        T[n][n] = 0.0
        T[:n, :n] = K + self.mu * np.eye(n)
        H = np.concatenate((K, np.ones([n, 1])), axis=1).dot(np.linalg.inv(T))
        loo_resid = (self.y - yhat) / (1 - H.diagonal())
        press = (loo_resid ** 2).sum()
        return loo_resid

    def PRESS(self):
        loo=self.loo_residuals()
        return loo.dot(loo)

    # finds optimal regularisation parameter
    def optim_reg_param(self, x, y, Mu=muArray):
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
            # print("Mu= %2.4f  PRESS=%f"%(Mu[i],PRESS[i]))
        return Mu[PRESS.argmin()], min(PRESS)

    def fullyOptimRBF(self,x, y):
        kn = RBF()
        kn.setInitWidh(self.x)
        sig = kn.getWidth()
        sigma = (10 ** np.arange(-3, 2.25, 0.25)) * sig
        muX = np.zeros(len(sigma))
        pressX = np.zeros(len(sigma))
        for i in range(len(sigma)):
            ls = lssvm(RBF(sigma[i]))
            muX[i], pressX[i] = ls.optim_reg_param(x, y)
            print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f" %
                  (sigma[i], muX[i], pressX[i]))
        muOpt = muX[pressX.argmin()]
        sigOpt = sigma[pressX.argmin()]
        print("Optimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" %
              (sigOpt, muOpt))
        netOpt = lssvm(RBF(sigOpt), muOpt)
        print("training with opt parameters...")
        netOpt.fit(self.x, self.y)
        return netOpt

    def __str__(self):
        return self.kernel.__str__() + "  Regularisation parameter = " + \
               str(self.mu)[:6]
