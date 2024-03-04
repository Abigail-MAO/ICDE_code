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
import sax_generate as sg
import test

# path = '/data/fat/maoyl/ShapeExtraction/'+'07_02'+'/patternLDP/'
path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/'


def read_data(file_path, label=False):
    file = open(file_path, 'r')
    data = []
    while True:
        line = file.readline()
        if not line:
            break
        if label:
            data.append(int(line))
        else:
            data.append(line.strip())
    file.close()
    return data


def clus_rate(para):
    epsilon, index = para[0], para[1]
    # print(epsilon, index)
    train_data = np.load(path+'train/patternLDP/'+str(epsilon)+'/'+str(index)+'_data.npy')
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
   
    ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
    ks_centers = []
    for i in range(6):
        ks_centers.append(sg.sax_delete_repeat(sg.sax(ks.cluster_centers_[i], 6, 25)))
    centers = ['abcdef', 'abcdef', 'babcdfed', 'dcadfcadfcacfec', 'ecadfcbfeadfcad', 'fedcbabcdef']
    dist, clus = test.match_clustering(ks_centers, centers)
    test_data = read_data('../data/'+'/test_data.txt')
    test_label = read_data('../data/'+'/test_label.txt', label=True)
    tp = test.true_positive(clus, test_data, test_label)
    print(tp)
    return (epsilon, tp)


def write_result(para):
    epsilon, dist = para[0], para[1]
    file = open('./patternLDP_result/'+str(epsilon)+"_result_tp.txt", "a+")
    file.write(str(dist)+"\n")
    file.close()


if __name__ == '__main__':
    epsilons = [4]
    
    folder = os.path.exists("./patternLDP_result/")
    if not folder:
        os.makedirs("./patternLDP_result/")
    # else:
    #     files = os.listdir("./result/")
    #     for f_path in files:
    #         os.remove("./result/"+f_path)
    para = []
    for epsilon in epsilons:
        for index in range(0, 28):
            para.append((epsilon, index))
    
    # clus_rate(para[0])
    # exit()
    # for i in range(100):
    #     clus_rate(para[i])
    core_num = 28
    pool = Pool(core_num)
    for i in range(len(para)):
            pool.apply_async(clus_rate, args=(para[i],), callback=write_result)
    pool.close()
    pool.join()
    
    # 结果在clab03上
    exit()
    
    # core_num = 28
    # pool = Pool(processes=core_num)
    # for epsilon in epsilons:
    #     para = []
    #     for index in range(0, 500):
    #         para.append((epsilon, index))
    # # for i in range(10):
    # #     result = clus_rate(para[i])
    # #     write_result(result)q
        
    #     for i in range(len(para)):
    #             pool.apply_async(clus_rate, args=(para[i],), callback=write_result)
    #     pool.close()
    #     pool.join()
    # exit()
    # os._exit(0)
    
    # /data/fat/maoyl/ShapeExtraction/07_02/patternLDP/0.5/0_data.npy