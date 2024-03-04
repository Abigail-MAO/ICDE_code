import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
import time
import os
from multiprocessing import Pool
from tslearn.clustering import KShape
import sys
from sklearn.metrics.cluster import adjusted_rand_score as ari
import time
import matplotlib.pyplot as plt
import time
import dtw
import matplotlib.pyplot as plt


def clus_rate(para):
    epsilon, index = para[0], para[1]
    # print(epsilon, index)
    train_data = np.load(str(index)+'_data.npy')
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    # test_data = np.load(str(index)+'_data.npy')
    test_data = np.load('../data/test_data.npy')[:, :len(train_data[0])]
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    test_label = np.load('../data/test_label.npy')
    
    ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
    centers = ks.cluster_centers_
    ground_truth = np.load('../ground_truth/ground_truth_center.npy')
    patternLDP_centers = []
    has_visited = [False for i in range(len(centers))]
    for i in range(len(centers)):
        dist = [dtw.dtw(ground_truth[i], centers[j]).distance for j in range(len(centers))]
        for j in range(len(dist)):
            if has_visited[j]:
                dist[j] = sys.maxsize
        min_index = np.argmin(dist)
        has_visited[min_index] = True
        patternLDP_centers.append(centers[min_index])
    patternLDP_centers = np.array(patternLDP_centers)
    for i in range(len(patternLDP_centers)):
        plt.plot(patternLDP_centers[i])
    plt.savefig('patternLDP_center.png')
    np.save('patternLDP_center.npy', patternLDP_centers)
    
    # exit()
    
    y_pred = ks.predict(test_data)
    acc = ari(test_label, y_pred)
    print(acc)
    return (epsilon, acc)


def write_result(para):
    epsilon, acc = para[0], para[1]
    file = open('./result/'+str(epsilon)+"_result.txt", "a+")
    file.write(str(acc)+"\n")
    file.close()


if __name__ == '__main__':
   
    epsilon = 4
    index = 0
    np.random.seed(2023)
    
    clus_rate(((epsilon, index)))