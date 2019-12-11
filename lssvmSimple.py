
import matplotlib.pyplot as plt
import numpy as np
from lssvm import lssvm
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
    plt.plot(net.x, net.yhat, '-r', label='Predict')
    plt.title("Initial Model, "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    #plt.show()

    print("\nfinding best regularisation parameter, initial RBF width fixed ")
    print("optimalRegularisation")
    net.optim_refit()
    print(net.mu)
    plt.subplot(312)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.yhat, '-r', label='Predict')
    plt.title("Opt Reg  with same RBF, "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    #plt.show()

    print("\nfull optimisation RBF width and regularisation parameter")
    net = net.fullyOptimRBF()
    plt.subplot(313)
    plt.plot(x, y, '.', label='Train')
    plt.plot(x, y0, 'k--', label='Exact')
    plt.plot(net.x, net.yhat, '-r', label='Predict')
    plt.title("Full optim , "+ net.__str__(),fontsize=10)
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def bin_class():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("banana.csv")

    df.loc[df['Class']==1,'Class'] = -1
    df.loc[df['Class']==2,'Class'] = 1
    target = df.pop('Class')

    #--- for graphing purpose
    pos = df[target.values > 0]
    neg = df[target.values < 0]
    v1_min, v1_max = df['V1'].min(), df['V1'].max()
    v2_min, v2_max = df['V2'].min(), df['V2'].max()
    v1, v2 = np.meshgrid(np.linspace(v1_min,v1_max,50),
                         np.linspace(v2_min, v2_max,50))
    s = np.stack( [v1.reshape(-1, 1).flatten(), v2.reshape(-1, 1).flatten()],
                  axis=1)

    def plot_contour(yhat, title, imageName=None):
        z = yhat.reshape(v1.shape[0], v2.shape[0])
        # fig, ax = plt.subplots(constrained_layout=True)
        # CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin=origin)
        fig, ax = plt.subplots(1, 1,figsize=(10,9))
        cp = ax.contourf(v1, v2, z,cmap=plt.cm.bone)
        #ax.contourf(cp, colors='k')
        fig.colorbar(cp)
        ax.set_title(title,fontsize=10)
        plt.plot(pos['V1'], pos['V2'], 'r.',label='Class +1')
        plt.plot(neg['V1'], neg['V2'], 'g.',label='Class -1')
        plt.legend()
       #plt.show()
        if imageName:
            plt.savefig(imageName)

    X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                        test_size=0.90,
                                                        random_state=42)

    net = lssvm(RBF(0.1), mu=0.1)

    net.fit(X_train.values,y_train.values)
    yhat = net.predict(s)
    plot_contour(yhat, "Initial model: " + net.__str__(), "initMod")

    net.optim_refit()
    yhat = net.predict(s)
    plot_contour(yhat, "Optim Regularised Model: " + net.__str__(), "optimReg")

    net = net.fullyOptimRBF()
    yhat = net.predict(s)
    plot_contour(yhat, "Full optimised Model: " + net.__str__(), "fullyoptm")



if __name__=="__main__":
    bin_class()
    #regression()




























