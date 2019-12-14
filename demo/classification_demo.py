import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lssvm.lssvm import lssvm
from kernels.rbf import RBF


def bin_class():

    df = pd.read_csv("data/banana.csv")
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

    net.fit(X_train.values, y_train.values)
    yhat = net.predict(s)
    plot_contour(yhat, "Initial model: " + net.__str__(), "initMod")

    mu, _ = net.optim_reg_param(X_train.values, y_train.values)
    net.mu = mu
    net.fit(X_train.values, y_train.values)
    yhat = net.predict(s)
    plot_contour(yhat, "Optim Regularised Model: " + net.__str__(), "optimReg")

    net = net.fullyOptimRBF(X_train.values, y_train.values)
    yhat = net.predict(s)
    plot_contour(yhat, "Full optimised Model: " + net.__str__(), "fullyoptm")

if __name__=="__main__":
    bin_class()

