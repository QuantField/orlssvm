
import numpy as np
import matplotlib.pyplot as plt
import abc
import copy


class Kernel:
    def __init__(self, kerntype):
        self._type = "Not defined"
        s = kerntype.strip().upper()
        if s in ('LINEAR', 'POLYNOMIAL', 'RBF'):
            self._type = s
        else:
            print("invalid Kernel, only  LINEAR,POLYNOMIAL,RBF are allowed")

    @abc.abstractmethod
    def params(self):
        return self._type

    def __str__(self):
        return self.params()

    @abc.abstractmethod
    def evaluate(self, x1, x2):
        return


class Linear(Kernel):
    def __init__(self):
        Kernel.__init__(self, 'Linear')

    def evaluate(self, x1, x2):
        return x2.dot(x1.T)


class Polynomial(Kernel):
    def __init__(self, order, ofset):
        super(Polynomial, self).__init__('Polynomial')
        self.__order = order
        self.__ofset = ofset

    def evaluate(self, x1, x2):
        return (x2.dot(x1.T) + self.__ofset) ** self.__order

    def params(self):
        s = "Order = " + str(self.__order) + "   Ofset = " + str(self.__ofset)
        return s


class RBF(Kernel):

    def __init__(self, sigma=0.5):
        super(RBF, self).__init__('RBF')
        self.__width = sigma
        self.__type = 'RBF'

    def width(self):
        return self.__width

    # Good starting point for the width
    def setInitWidh(self, trData):
        self.__width = 0.5 * np.linalg.norm(trData.std(0))

    def getWidth(self):
        return self.__width

    def squared_distance(self, x1, x2):
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
        w = (self.__width) ** 2
        K = self.squared_distance(x1, x2)
        return np.exp(-K / w).T

    def params(self):
        return 'RBF Width = ' + str(self.__width)[:6]


class lssvm:
    muArray = np.logspace(-3, 2, 50)

    def __init__(self, kern=RBF(), mu=0.1):
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
        self.ntp = n;
        self.x = x
        self.y = y
        K = self.Kernel.evaluate(x, x)
        T = np.ones([n + 1, n + 1]);
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
            K = self.Kernel.evaluate(xt[list(range(start, stop + 1)), :], self.x)
            prd = (K.T).dot(self.alpha) + self.bias
            pred = np.append(pred, prd)
        return pred

    def residuals(self, y, yhat):
        dy = (y - yhat)
        r2 = 1 - sum(dy ** 2) / sum((y - np.mean(y)) ** 2)
        return dy, r2

    def looResiduals(self):
        n = self.ntp;
        K = self.Kernel.evaluate(self.x, self.x);
        T = np.ones([n + 1, n + 1]);
        T[n][n] = 0.0
        T[:n, :n] = K + self.mu * np.eye(n)
        H = np.concatenate((K, np.ones([n, 1])), axis=1).dot(np.linalg,inv(T))
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
        xi2 = xi ** 2;
        PRESS = np.zeros(len(Mu))
        for i in range(len(Mu)):
            u = xi / (eigVal + Mu[i])
            g = eigVal / (eigVal + Mu[i])
            sm = -(xi2 / (eigVal + Mu[i])).sum()
            theta = Vt_y / (eigVal + Mu[i]) + (u.dot(Vt_y) / sm) * u
            h = Vt_sqr.T.dot(g) + (V.dot(u * eigVal) - 1) * (V.dot(u)) / sm
            f = V.dot(eigVal * theta) - sum(u * Vt_y) / sm
            loo_resid = (y - f) / (1 - h);
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
        kn = RBF()
        kn.setInitWidh(self.x)
        sig = kn.getWidth()
        sigma = (10 ** np.arange(-3, 2.25, 0.25)) * sig
        muX = np.zeros(len(sigma))
        pressX = np.zeros(len(sigma))
        for i in range(len(sigma)):
            ls = lssvm(RBF(sigma[i]))
            ls.setXY(self.x, self.y)
            muX[i], pressX[i] = ls.get_optim_regparam()
            print("Width = %4.4f  Mu =%4.4f  PRESS=%8.4f" % (sigma[i], muX[i], pressX[i]))
        muOpt = muX[pressX.argmin()]
        sigOpt = sigma[pressX.argmin()]
        print("Optimal Parameters: RBF Width =%4.6f, Regular Param =%4.6f" % (sigOpt, muOpt))
        netOpt = lssvm(RBF(sigOpt), muOpt)
        print("training with opt parameters...")
        netOpt.fit(self.x, self.y)
        return netOpt

    def __str__(self):
        return self.Kernel.__str__() + "  Regularisation parameter = "+ str(self.mu)[:6]


# --------------------------------------------------------------------------------------------


x0 = np.linspace(-10, 10, 200)
y0 = np.sin(x0) / x0

# plt.plot(x0,y0,'r-')
x = x0
y = y0 + np.random.normal(0, 0.2, len(x0))
# plt.plot(x,y,'go')

net = lssvm(RBF(), mu=0.1)


# x.shape = (len(x),1)
print("\n---------- Initial Model -----------")
print("Training.....")
net.fit(x.reshape(-1, 1), y)
plt.figure(figsize=(8,8))
plt.subplot(311)
plt.subplots_adjust(hspace = 0.3)
plt.plot(x, y, '.', label='Train')
plt.plot(x, y0, 'k--', label='Exact')
plt.plot(net.x, net.yhat, '-r', label='Predict')
plt.title("Initial Model, "+ net.__str__())
plt.legend(loc='upper right', shadow=True)
#plt.show()

print("\n---------- finding best regularisation parameter, initial RBF width fixed -----------")
print("optimalRegularisation")
net.optim_refit()
print(net.mu)
plt.subplot(312)
plt.plot(x, y, '.', label='Train')
plt.plot(x, y0, 'k--', label='Exact')
plt.plot(net.x, net.yhat, '-r', label='Predict')
plt.title("Opt Reg  with same RBF, "+ net.__str__())
plt.legend(loc='upper right', shadow=True)
#plt.show()

print("\nfull optimisation RBF width and regularisation parameter")
net = net.fullyOptimRBF()
plt.subplot(313)
plt.plot(x, y, '.', label='Train')
plt.plot(x, y0, 'k--', label='Exact')
plt.plot(net.x, net.yhat, '-r', label='Predict')
plt.title("Full optim , "+ net.__str__())
plt.legend(loc='upper right', shadow=True)
plt.show()




