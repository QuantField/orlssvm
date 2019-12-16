import numpy as np
from  kernels.rbf import RBF
import copy

class lssvm:

    def __init__(self, kern = RBF(), mu=0.1):
        self.alpha = None
        self.ntp = 0.0
        self.bias = 0.0
        self.x = None
        self.y = None
        self.mu = mu
        self.kernel = kern

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
        return loo_resid

    def press(self):
        loo_resid=self.loo_residuals()
        return loo_resid.dot(loo_resid)

    def loo_error(self):
        # for this to work correctly y must be in {-1,1}
        loo_resid = self.loo_residuals()
        return np.mean(self.y*loo_resid-1>0)

    def __str__(self):
           return self.kernel.__str__() + \
                  "  Regularisation parameter = " + \
                  str(self.mu)[:6]
