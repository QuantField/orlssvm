import sys 
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from optlssvm.kernels import RBF
from optlssvm.lssvm import LSSVM
from optlssvm.orlssvm import OptimallyRegularizedLSSVM
from optlssvm.opt_rbf_lssvm import OptimallyRegularizedRBFLSSVM


def main():
    x0 = np.linspace(-10, 10, 200)
    y0 = np.sin(x0) / x0
    x = x0
    y = y0 + np.random.normal(0, 0.2, len(x0))
    net = LSSVM(RBF(), mu=0.1)

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
    # plt.show()

    print("\nfinding best regularisation parameter, initial RBF width fixed ")
    print("optimalRegularisation")
    net = OptimallyRegularizedLSSVM(RBF())
    net.fit(x.reshape(-1, 1), y)
    plt.subplot(312)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.predict(x.reshape(-1,1)), '-r', label='Predict')
    plt.title("Opt Reg  with same RBF, "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    # plt.show()

    print("\nfull optimisation RBF width and regularisation parameter")
    net = OptimallyRegularizedRBFLSSVM()
    net.fit(x.reshape(-1,1),y)
    plt.subplot(313)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.predict(x.reshape(-1,1)), '-r', label='Predict')
    plt.title("Full optim , "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    plt.show()

if __name__=='__main__':
    main()
