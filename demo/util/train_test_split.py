import pandas as pd 
import numpy as np 

def train_test_split(df, target, train_frac, random_state = 123):
    np.random.seed(random_state)
    train_ind = np.random.rand(len(df)) < train_frac 
    x_train, y_train = df[train_ind], target[train_ind]
    x_test,  y_test  = df[~train_ind], target[~train_ind]
    return  x_train, x_test, y_train, y_test

