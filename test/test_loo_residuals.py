import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from optlssvm.lssvm import LSSVM

ZERO = 1.0E-8
# loading banana dataset
df = pd.read_csv("../data/banana.csv")
n = 200
df = df.sample(n)
# binary classification, target/(class label) in LSSVM is in {-1,1}
df.loc[df['Class']==1,'Class'] = -1
df.loc[df['Class']==2,'Class'] = 1

# calculating via the conventional way
net = LSSVM()
all = list(range(n))
loo_residuals = np.zeros([n,1])
loo_error     = np.zeros([n,1])
for i in range(n):
    loo_index = all.copy()
    loo_index.remove(i)
    xtrain = df.iloc[loo_index,:]
    ytrain = xtrain.pop('Class')
    xtest  = df.iloc[[i],:]
    ytest  = xtest.pop('Class')
    net.fit(xtrain.values, ytrain.values)
    loo_residuals[i] = float(ytest) - float(net.predict(xtest.values))
    loo_error[i] = np.sign(ytest) != np.sign(net.predict(xtest.values))

# calculating using closded form
y = df.pop('Class') #
net.fit(df.values, y) # training

loo_residuals_method = net.loo_residuals()
loo_err_method = net.loo_error()

def test_loo_residuals():
    diff = loo_residuals_method - loo_residuals.reshape(-1, )
    assert np.max(np.abs(diff)) <= ZERO

def test_loo_error():
    assert np.mean(loo_error) == loo_err_method