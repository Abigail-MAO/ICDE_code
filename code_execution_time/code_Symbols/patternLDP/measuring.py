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
import warnings

# path = '/data/fat/maoyl/ShapeExtraction/'+'07_02'+'/patternLDP/'
# path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/'
path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/train/'

def clus_rate(para):
    epsilon, index = para[0], para[1]
    np.random.seed(2)
    # print(epsilon, index)
    train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    # test_data = np.load(path+'test/patternLDP/'+str(epsilon)+'/'+str(index)+'_data.npy')
    test_data = np.load('../data/test_data.npy')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    test_data = test_data[:, :len(train_data[0])]
    test_label = np.load('../data/test_label.npy')
    # print(train_data.shape)
    # for i in range(len(train_data[:10])):
    #     plt.plot(train_data[i])
    # plt.savefig('test'+str(i)+'.png')
    # # print(test_label[1])
    # # print(train_data.shape, test_data.shape, test_label.shape)
    # return
    # print(len(train_data))
    # start_time = time.time()
    # ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
    ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False).fit(train_data)
    # print(ks.cluster_centers_)
    # file = open('../cluster_result/patternLDP.npy', 'wb')
    # np.save(file, ks.cluster_centers_)
    # file.close()
    # sys.exit()
    
    # sys.exit()
    y_pred = ks.predict(test_data)
    acc = ari(test_label, y_pred)
    # print(acc)
    return (epsilon, acc)


def write_result(para):
    epsilon, acc = para[0], para[1]
    file = open('./result/'+str(epsilon)+"_result.txt", "a+")
    file.write(str(acc)+"\n")
    file.close()


if __name__ == '__main__':
    # train_data = np.load('../data/train_data.npy')
    # test_data = np.load('../data/test_data.npy')
    # test_label = np.load('../data/test_label.npy')
    # ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
    # y_pred = ks.predict(test_data)
    # acc = ari(test_label, y_pred)
    # print(acc)
    # exit()
    
    # sys.exit()
    # epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    # epsilons = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    # epsilons = [6, 6.5, 7, 7.5, 8]
    warnings.filterwarnings("ignore")
    # , 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    epsilons = [12]
    print(epsilons)
    
    folder = os.path.exists("./result/")
    if not folder:
        os.makedirs("./result/")
    # else:
    #     files = os.listdir("./result/")
    #     for f_path in files:
    #         os.remove("./result/"+f_path)
    para = []
    for epsilon in epsilons:
        for index in range(0, 500):
            para.append((epsilon, index))
    # print(clus_rate(para[0]))
    # exit()
    # start_time = time.time()
    # for i in range(100):
    #     clus_rate(para[i])
    # print((time.time()-start_time)/100)
    
    # 9.979701490402222
    
    
    core_num = 20
    pool = Pool(core_num)
    for i in range(len(para)):
            pool.apply_async(clus_rate, args=(para[i],), callback=write_result)
    pool.close()
    pool.join()
    
    
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