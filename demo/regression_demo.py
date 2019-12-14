import matplotlib.pyplot as plt
import numpy as np
from lssvm.lssvm import lssvm
from kernels.rbf import RBF

def regression():
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
    plt.figure(figsize=(8,10))
    plt.subplot(311)
    plt.subplots_adjust(hspace = 0.3)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.predict(x.reshape(-1,1)), '-r', label='Predict')
    plt.title("Initial Model, "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    #plt.show()

    print("\nfinding best regularisation parameter, initial RBF width fixed ")
    print("optimalRegularisation")
    mu, _ = net.optim_reg_param(x.reshape(-1, 1), y)
    net.mu = mu
    net.fit(x.reshape(-1, 1), y)

    print(net.mu)
    plt.subplot(312)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.predict(x.reshape(-1,1)), '-r', label='Predict')
    plt.title("Opt Reg  with same RBF, "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    #plt.show()

    print("\nfull optimisation RBF width and regularisation parameter")
    net = net.fullyOptimRBF(x.reshape(-1,1),y)
    plt.subplot(313)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.predict(x.reshape(-1,1)), '-r', label='Predict')
    plt.title("Full optim , "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    plt.show()

if __name__=='__main__':
    regression()