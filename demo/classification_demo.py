import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lssvm.lssvm import lssvm
from lssvm.or_lssvm import or_lssvm
from lssvm.opt_rbf_lssvm import opt_rbf_lssvm
from kernels.rbf import RBF

# loading banana dataset
df = pd.read_csv("../data/banana.csv")
# binary classification, target/(class label) in LSSVM is in {-1,1}
df.loc[df['Class']==1,'Class'] = -1
df.loc[df['Class']==2,'Class'] = 1
#------------------------
target = df.pop('Class')
#------------------------

pos = df[target.values > 0]
neg = df[target.values < 0]

#--- for graphing purpose
def meshgrid_2d_test_data(df, var1, var2, target):
    n_grid_points = 50
    v1_min, v1_max = df[var1].min(), df[var1].max()
    v2_min, v2_max = df[var2].min(), df[var2].max()
    v1, v2 = np.meshgrid(np.linspace(v1_min, v1_max, n_grid_points),
                         np.linspace(v2_min, v2_max, n_grid_points))
    s = np.stack( [v1.reshape(-1, 1).flatten(),
                   v2.reshape(-1, 1).flatten()], axis=1)
    return s, v1, v2


def plot_contour(yhat,v1,v2, title, imageName=None):
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


xplan, v1, v2  = meshgrid_2d_test_data(df,'V1','V2', target)


X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                    test_size=0.90,
                                                    random_state=42)
print("LSSVM:")
net = lssvm(RBF(0.1), mu=0.1)
# training
net.fit(X_train.values, y_train.values)
# scoring
yhat = net.predict(xplan)
plot_contour(yhat, v1,v2, "Initial model: " + net.__str__(), "initMod")

print("\n Optimally Regularisded LSSVM")
net = or_lssvm(net.kernel)
net.fit(X_train.values, y_train.values)
yhat = net.predict(xplan)
plot_contour(yhat, v1,v2, "Initial model: " + net.__str__(), "optReg")

print("\n Optimal RBF LSSVM, both RBF width and regularisation parameter are\
are optimised")
net = opt_rbf_lssvm()
net.fit(X_train.values, y_train.values)
yhat = net.predict(xplan)
plot_contour(yhat, v1,v2, "Initial model: " + net.__str__(), "fullOpt")


plt.show()