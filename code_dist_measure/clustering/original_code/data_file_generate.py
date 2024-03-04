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


def read_csv(file_name):
    label_dict = {'class 1': 1, 'class 2': 2, 'class 3': 3, 'class 4': 4, 'class 5': 5, 'class 6':6}
    file = open('../data/'+file_name+'.csv')
    lines = file.readlines()
    file.close()
    label = []
    data = []
    for line in lines:
        d = line.strip().split(',')
        d.remove('')
        # print(d)
        l = label_dict[d[-1]]
        d = list(map(lambda x: float(x), d[0:len(d)-1]))
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
    print(len(data), len(data[0]), len(label))

def data_load():
    f = signal.butter(5, 10, 'lp', fs=50, output='sos')
    file = open('../data/Symbols_TEST.txt')
    lines = file.readlines()
    file.close()
    label = []
    data = []
    for line in lines:
        d = line.strip().split('  ')
        l = int(float(d[0]))
        if l==-1:
            l = 2
        # if l>2:
        #     continue
        d = list(map(lambda x: float(x), d[1:]))
        # d = signal.sosfilt(f, d)
        # d = stats.zscore(d)
        label.append(l)
        data.append(d)
    file = open('../data/Symbols_TRAIN.txt')
    lines = file.readlines()
    file.close()
    # label = []
    # data = []
    for line in lines:
        d = line.strip().split('  ')
        l = int(float(d[0]))
        if l==-1:
            l = 2
        # if l>2:
        #     continue
        d = list(map(lambda x: float(x), d[1:]))
        # d = signal.sosfilt(f, d)
        # d = stats.zscore(d)
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
    
    # count = 10
    # index1 = 0
    # index2 = 0
    # # index3 = 0
    # # index4 = 0
    # i = 0
    # # print(label[:10])
    # while index2<count or index1<count :
    #     if label[i] == 1 and index1<count:
    #         plt.plot(data[i], color='red')
    #         index1 += 1
    #     if label[i] == 2 and index2<count:
    #         plt.plot(data[i], color='blue')
    #         index2 += 1
    # #     # elif label[i] == 3 and index3<count:
    # #     #     plt.plot(data[i], color='green')
    # #     #     index3 += 1
    # #     # elif label[i] == 4 and index4<count:
    # #     #     plt.plot(data[i], color='black')
    # #     #     index4 += 1
    #     i += 1
    # plt.savefig('../data/plot.png')
    

def data_augementation(data, label):
    for i in range(0):
        data = np.concatenate((data, np.roll(data, i+1, axis=1)), axis=0)
        label = np.concatenate((label, label), axis=0)
    print(len(data), len(label))
    return data, label

def sax_generate_files(data, label, symbol_size, length):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=1000, random_state=2023)
    file_train_data = open('../data/train_data.npy', 'wb')
    file_train_label = open('../data/train_label.npy', 'wb')
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    np.save(file_train_data, X_train)
    np.save(file_train_label, y_train)
    file_test_data = open('../data/test_data.npy', 'wb')
    file_test_label = open('../data/test_label.npy', 'wb')
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
    np.save(file_test_data, X_test)
    np.save(file_test_label, y_test)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # file_data = open('../data/data.txt', 'a+')
    # file_label = open('../data/label.txt', 'a+')
    # for i in range(X_train.shape[0]):
    #     sax = sg.sax_delete_repeat(sg.sax(X_train[i], symbol_size, length))
    #     # print(sax)
    #     # sax_list = sg.sax(X_train[i], symbol_size, length)
    #     # sax = ''
    #     # for s in sax_list:
    #     #     sax += s
    #     file_data.write(sax+'\n')
    #     file_label.write(str(y_train[i])+'\n')
    # file_data.close()
    # file_label.close()
    # file_data = open('../data/test_data.txt', 'a+')
    # file_label = open('../data/test_label.txt', 'a+')
    # for i in range(X_test.shape[0]):
    #     sax = sg.sax_delete_repeat(sg.sax(X_test[i], symbol_size, length))
    #     # sax_list = sg.sax(X_test[i], symbol_size, length)
    #     # sax = ''
    #     # for s in sax_list:
    #     #     sax += s
    #     file_data.write(sax+'\n')
    #     file_label.write(str(y_test[i])+'\n')
    # file_data.close()
    # file_label.close()
    # pass
    
    
if __name__ == '__main__':
    # read_csv('new_data')
    # # print('data_load done')
    data = np.load('../data/data.npy')
    label = np.load('../data/label.npy')
    # # print(data.shape , label.shape)
    # data, label = data_augementation(data, label)
    symbol_size = 6
    length = 25
    sax_generate_files(data, label, symbol_size, length)
    
    # from pyts.approximation import SymbolicFourierApproximation
    # transformer = SymbolicFourierApproximation(n_coefs=4, n_bins=4, strategy='quantile')
    # X_new = transformer.fit_transform(data)
    # print(X_new.shape)
    
    # for l in range(1, 3):
    #     count=0
    #     i=0
    #     while count<10:
    #         if label[i] == l:
    #             print(X_new[i], label[i])
    #             count += 1
    #         i += 1
    #     print()
    
    
    
    
    
    