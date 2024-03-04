from tslearn.svm import TimeSeriesSVC
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import time
import os
from multiprocessing import Pool
import sys
from sklearn.neighbors import KNeighborsClassifier
import time
import random
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import dtw
import warnings


def class_rate(para):
    epsilon, index = para
    train_data = np.load(str(index)+'_data_e8.npy')
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    train_label = np.load(str(index)+'_label_e8.npy')
    test_data = np.load('../data/test_data.npy')[:, :297]
    test_label = np.load('../data/test_label.npy')
    
    indicies = np.random.choice(train_data.shape[0], 300, replace=False)
    selected_data = train_data[indicies]
    ks = KShape(n_clusters=3, random_state=2023).fit(selected_data)
    centers = ks.cluster_centers_
    ground_truth = np.load('../ground_truth/ground_truth_center.npy')
    patternLDP_centers = []
    has_visited = [False for i in range(len(centers))]
    for i in range(len(centers)-1, -1, -1):
        dist = [dtw.dtw(ground_truth[i], centers[j]).distance for j in range(len(centers))]
        for j in range(len(dist)):
            if has_visited[j]:
                dist[j] = sys.maxsize
        min_index = np.argmin(dist)
        has_visited[min_index] = True
        patternLDP_centers.append(centers[min_index])
    patternLDP_centers = np.array(patternLDP_centers)
    np.save('patternLDP_center_e8.npy', patternLDP_centers)
    # exit()
    
    
    clf = RandomForestClassifier()
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print(acc)
    return (epsilon, acc)



if __name__ == '__main__':
    epsilon = 8
    index = 0
    class_rate((epsilon, index))
    
    
    
    
    
    