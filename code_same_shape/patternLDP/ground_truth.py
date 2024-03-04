from tslearn.svm import TimeSeriesSVC
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import time
import os
from multiprocessing import Pool
import sys
from _fast_ddtw import *
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split

path = '/data/fat/maoyl/ShapeExtraction/'+'06_02'+'/patternLDP/'


def ddtw_1nn(para):
    epsilon, index = para
    print(epsilon)
    train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')[:1000]
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    train_label = np.load(path+str(epsilon)+'/'+str(index)+'_label.npy')[:1000]
    test_data = np.load('../data/test_data.npy')[:, :-1]
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    test_label = np.load('../data/test_label.npy')
    # print(len(train_data), len(test_data))
    # print("read data done")
    
    start_time = time.time()
    neigh = KNeighborsClassifier(n_neighbors=1, metric=fast_ddtw, n_jobs=-1)
    neigh.fit(train_data, train_label)
    print("classifier done")
    y_pred = neigh.predict(test_data)
    print("predict done")
    acc = neigh.score(test_data, test_label)
    end_time = time.time()
    print(end_time-start_time)
    return acc
    
    # print("distance matrix done")
    
    

def class_rate(para):
    epsilon, index = para
    print(epsilon)
    
    train_data = np.load('../data/train_data.npy')
    train_label = np.load('../data/train_label.npy')
    test_data = np.load('../data/test_data.npy')
    test_label = np.load('../data/test_label.npy')
    
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    
    # sys.exit()
    clf = RandomForestClassifier(min_samples_leaf=20)
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print(acc)
    return (epsilon, acc)

def write_result(para):
    epsilon, acc = para[0], para[1]
    file = open('./result/'+str(epsilon)+"_result.txt", "a+")
    file.write(str(acc)+"\n")
    file.close()

if __name__ == '__main__':
    class_rate((0.1, 0))
   