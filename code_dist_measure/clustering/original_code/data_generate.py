import sax_generate as sg
from tslearn.datasets import UCR_UEA_datasets
import os
import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import scipy.signal as signal
import scipy.stats as stats
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import PiecewiseAggregateApproximation


def data_load():
    file = open('../data/ECG200_TEST.txt')
    lines = file.readlines()
    file.close()
    label = []
    data = []
    for line in lines:
        d = line.strip().split('  ')
        l = int(float(d[0]))
        d = list(map(lambda x: float(x), d[1:]))
        label.append(l)
        data.append(d)
    file = open('../data/ECG200_TRAIN.txt')
    lines = file.readlines()
    file.close()
    for line in lines:
        d = line.strip().split('  ')
        l = int(float(d[0]))
        d = list(map(lambda x: float(x), d[1:]))
        label.append(l)
        data.append(d)
    data = np.array(data)
    label = np.array(label)
    file = open('../data/data.npy', 'wb')
    np.save(file, data)
    file.close()
    file = open('../data/label.npy', 'wb')
    np.save(file, label)
    file.close()
    print(len(data), len(label))
    print(set(label))


def data_augementation(data, label):
    for i in range(9):
        data = np.concatenate((data, np.roll(data, i+1, axis=1)), axis=0)
        label = np.concatenate((label, label), axis=0)
    print(len(data), len(label))
    return data, label


def sax_generate_files(data, label, symbol_size, length):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=2023)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], -1)
    paa_transformer = PiecewiseAggregateApproximation(window_size=length)
    data = paa_transformer.transform(X_train)
    sax_transformer = SymbolicAggregateApproximation(n_bins=symbol_size, strategy='normal')
    data = sax_transformer.transform(data)
    file_data = open('../data/data.txt', 'a+')
    file_label = open('../data/label.txt', 'a+')
    word_length = len(data[0])
    for i in range(X_train.shape[0]):
        for j in range(word_length):
            file_data.write(str(data[i][j]))
        file_data.write('\n')
        file_label.write(str(y_train[i])+'\n')
    file_data.close()
    file_label.close()
    
    test_data = X_test.reshape(X_test.shape[0], -1)
    test_data = paa_transformer.transform(test_data)
    test_data = sax_transformer.transform(test_data)
    file_data = open('../data/test_data.txt', 'a+')
    file_label = open('../data/test_label.txt', 'a+')
    for i in range(X_test.shape[0]):
        for j in range(word_length):
            file_data.write(str(test_data[i][j]))
        file_data.write('\n')
        file_label.write(str(y_test[i])+'\n')
    file_data.close()
    file_label.close()
    
if __name__ == '__main__':
    data_load()
    # # print('data_load done')
    data = np.load('../data/data.npy')
    label = np.load('../data/label.npy')
    # print(data.shape , label.shape)
    data, label = data_augementation(data, label)
    symbol_size = 4
    length = 12
    sax_generate_files(data, label, symbol_size, length)
    