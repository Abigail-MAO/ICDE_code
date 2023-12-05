import numpy as np
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets as urc_d


def load_ECG():
    data_loader = urc_d()
    X_train, y_train, X_test, y_test = data_loader.load_dataset('Symbols')
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    file = open('./data/Symbols.csv', 'w')
    for i in range(len(X)):
        for j in range(len(X[i])):
            file.write(str(X[i][j][0])+',')
        file.write(str(y[i])+'\n')
    file.close()
    

if __name__=='__main__':
    load_ECG()