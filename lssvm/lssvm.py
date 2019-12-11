import numpy as np
import kernels as kern
import copy

class lssvm:

    muArray = np.logspace(-3, 2, 50)

    def __init__(self, kern = kern.RBF(), mu=0.1):
        self.alpha = None
        self.ntp = 0.0
        self.bias = 0.0
        self.x = None
        self.y = None
        self.mu = mu
        self.Kernel = kern
        self.yhat = None

    def copy(self):
        return copy.deepcopy(self)

    def fit(self, x, y):
        n = len(y)
        self.ntp = n
        self.x = x
        self.y = y
        K = self.Kernel.evaluate(x, x)
        T = np.ones([n + 1, n + 1])
        T[n][n] = 0.0
        T[:n, :n] = K + self.mu * np.eye(n)
        Sol = np.linalg.solve(T, np.append(y, 0))
        self.alpha, self.bias = Sol[0:n], Sol[n]
        self.yhat = K.dot(self.alpha) + self.bias
        print("Rsquared =", self.residuals(y, self.yhat)[1])

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
            K = self.Kernel.evaluate(xt[chunk_index, :], self.x)
            prd = (K.T).dot(self.alpha) + self.bias
            pred = np.append(pred, prd)
        return pred

    def residuals(self, y, yhat):
        dy = (y - yhat)
        r2 = 1 - sum(dy ** 2) / sum((y - np.mean(y)) ** 2)
        return dy, r2

    def looResiduals(self):
        n = self.ntp
        K = self.Kernel.evaluate(self.x, self.x)
        T = np.ones([n + 1, n + 1])
        T[n][n] = 0.0
        T[:n, :n] = K + self.mu * np.eye(n)
        H = np.concatenate((K, np.ones([n, 1])), axis=1)\
            .dot(np.linalg.inv(T))
        looResid = (self.y - self.yhat) / (1 - H.diagonal())
        press = (looResid ** 2).sum()
        return looResid, press

    # finds optimal regularisation parameter
    def get_optim_regparam(self, Mu=muArray):
        y = self.y
        eigVal, V = np.linalg.eigh(self.Kernel.evaluate(self.x, self.x))
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

    def optim_refit(self):
        print("tuning optimal regularisation parameter...")
        self.mu = self.get_optim_regparam()[0]
        self.fit(self.x, self.y)

    def setXY(self, x, y):
        self.x = x
        self.y = y
        self.ntp = len(y)

    def fullyOptimRBF(self):
        kn = kern.RBF()
        kn.setInitWidh(self.x)
        sig = kn.getWidth()
        sigma = (10 ** np.arange(-3, 2.25, 0.25)) * sig
        muX = np.zeros(len(sigma))
        pressX = np.zeros(len(sigma))
        for i in range(len(sigma)):
            ls = lssvm(kern.RBF(sigma[i]))
            ls.setXY(self.x, self.y)
            muX[i], pressX[i] = ls.get_optim_regparam()
            print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f" %
                  (sigma[i], muX[i], pressX[i]))
        muOpt = muX[pressX.argmin()]
        sigOpt = sigma[pressX.argmin()]
        print("Optimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" %
              (sigOpt, muOpt))
        netOpt = lssvm(kern.RBF(sigOpt), muOpt)
        print("training with opt parameters...")
        netOpt.fit(self.x, self.y)
        return netOpt

    def __str__(self):
        return self.Kernel.__str__() + "  Regularisation parameter = "+ \
               str(self.mu)[:6]
