import sys
sys.path.append('../')
from optlssvm.util import GridDataVisual, train_test_split
import pandas as pd
from optlssvm.lssvm import LSSVM
from optlssvm.orlssvm import OptimallyRegularizedLSSVM
from optlssvm.opt_rbf_lssvm import OptimallyRegularizedRBFLSSVM
from optlssvm.kernels import RBF


def run_lssvm(X_train,y_train, gridtest ):
    print("LSSVM:")
    net = LSSVM(RBF(0.1), mu=0.1)
    # training
    net.fit(X_train.values, y_train.values)
    # scoring
    yhat = net.predict(gridtest.xy_grid)
    gridtest.plot_contour(yhat, "Initial model: " + net.__str__(), 
                            image_name = None ) 

def run_optimally_regularised_lssvm(X_train,y_train, gridtest):
    print("\n Optimally Regularisded LSSVM")
    net = OptimallyRegularizedLSSVM(RBF(0.1))
    net.fit(X_train.values, y_train.values)
    yhat = net.predict(gridtest.xy_grid)
    gridtest.plot_contour(yhat, "Optimally regularised model: " + net.__str__(),
                            image_name = None)

def run_fully_optimised_lssvm(X_train,y_train, gridtest):
    print("\n Optimal RBF LSSVM, both RBF width and regularisation parameter are\
    are optimised")
    net = OptimallyRegularizedRBFLSSVM()
    net.fit(X_train.values, y_train.values)
    yhat = net.predict(gridtest.xy_grid)
    gridtest.plot_contour(yhat, "Fully optimised model: " + net.__str__(),
                            image_name = None)



def main():
    # ----------------------- loading banana dataset ---------------------------
    # Consists of two variables V1 and V2, and a target Class
    #
    df = pd.read_csv("../data/banana.csv")
    # binary classification, target/(class label) in LSSVM is in {-1,1}
    df.loc[df['Class'] == 1, 'Class'] = -1
    df.loc[df['Class'] == 2, 'Class'] = 1
    # --------------------------------------------------------------------------
    target = df.pop('Class')

    # 2D mesh grid creation for visualisation
    gridtest = GridDataVisual(df, 'V1', 'V2', target)

    # train/test split, only 10% of the data is used for training, 530
    # datapoints
    # no need for testing points as the whole 2d plan is used for testing, see
    # gridtest object
    X_train, _ , y_train, _ = train_test_split(df, target,train_frac=0.10,random_state=42)
    run_lssvm(X_train, y_train, gridtest)
    run_optimally_regularised_lssvm(X_train, y_train, gridtest)
    run_fully_optimised_lssvm(X_train, y_train, gridtest)

    gridtest.plt_show()


if __name__ == '__main__':
    main()

