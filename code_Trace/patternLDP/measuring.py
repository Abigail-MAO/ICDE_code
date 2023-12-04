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

# path = '/data/fat/maoyl/ShapeExtraction/'+'07_09'+'/patternLDP/'
path = '/data/fat/maoyl/ShapeExtraction/'+'12_04'+'/patternLDP/'
path_test = '/data/fat/maoyl/ShapeExtraction/'+'12_04_test'+'/patternLDP/'


def ddtw_1nn(para):
    epsilon, index = para
    print(epsilon)
    train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')[:1000]
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    train_label = np.load(path+str(epsilon)+'/'+str(index)+'_label.npy')[:1000]
    test_data = np.load(path_test+str(epsilon)+'/'+str(index)+'_data.npy')
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
    epsilon, index, test_data, test_label = para
    print(epsilon)
    train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    # print(train_data.shape)
    train_label = np.load(path+str(epsilon)+'/'+str(index)+'_label.npy')
    
    # print(len(train_data), len(test_data))
    # sys.exit()
    clf = RandomForestClassifier(min_samples_leaf=100)
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print(acc)
    return (epsilon, acc)

def write_result(para):
    epsilon, acc = para[0], para[1]
    file = open('./result_12_04/'+str(epsilon)+"_result.txt", "a+")
    file.write(str(acc)+"\n")
    file.close()

if __name__ == '__main__':
    # train = np.load('../data/train_data.npy')
    # print(train.shape)
    # train = train.reshape(train.shape[0], -1)
    # print(train.shape)
    # plt.plot(train[0])
    # plt.savefig("test.png")
    # train_label = np.load('../data/train_label.npy')
    # test = np.load('../data/test_data.npy')
    # test = test.reshape(test.shape[0], -1)
    # test_label = np.load('../data/test_label.npy')
    # start = time.time()
    # clf = RandomForestClassifier(min_samples_leaf=100)
    # clf.fit(train, train_label)
    # acc = clf.score(test, test_label)
    # print(acc)
    
    # print(time.time()-start)
    # sys.exit()
    
    
    test_data = np.load('../data/test_data.npy')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])[:, :297]
    test_label = np.load('../data/test_label.npy')
    # epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    # epsilons = [6.5, 7, 7.5, 8]
    epsilons = [7.5, 8]
    print(epsilons)
    para = []
    folder = os.path.exists("./result_12_04/")
    if not folder:
        os.makedirs("./result_12_04/")
    # else:
    #     files = os.listdir("./result/")
    #     for f_path in files:
    #         os.remove("./result/"+f_path)
    for epsilon in epsilons:
        for index in range(0, 100):
            para.append((epsilon, index, test_data, test_label))
    # for i in range(10):
    #     class_rate(para[i])
    core_num = 28
    pool = Pool(core_num)
    for i in range(len(para)):
            pool.apply_async(class_rate, args=(para[i],), callback=write_result)
    pool.close()
    pool.join()
    exit()
    # os._exit(0)
    
    
    
    # epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    # for epsilon in epsilons:
    #     file = open('./result/'+str(epsilon)+"_result.txt", "r")
    #     lines = file.readlines()
    #     acc = 0
    #     for line in lines:
    #         acc += float(line)
    #     acc /= len(lines)
    #     print(epsilon, acc)
    #     file.close()
    
    # epsilons = [8]
    # para = []
    # for epsilon in epsilons:
    #     for index in range(0, 1000):
    #         para.append((epsilon, index))
    # # for i in range(10):
    # acc = ddtw_1nn(para[0])
    # print(acc)
    # epsilon = 4
    # index = 0
    # # train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')
    # # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    # # train_label = np.load(path+str(epsilon)+'/'+str(index)+'_label.npy')
    # train_label = np.load('../data/train_label.npy')
    # train_data = np.load('../data/train_data.npy')[:, :-1]
    # result_index = [0, 0, 0]
    # label = [True, True, True]
    # for i in range(len(train_label)):
    #     if label[train_label[i]-1]:
    #         result_index[train_label[i]-1] = i
    #         label[train_label[i]-1] = False
    #     if True not in label:
    #         break
    # result = [train_data[result_index[i]] for i in range(3)]
    # result = np.array(result)
    # file = open('./class_data.npy', 'wb')
    # np.save(file, result)
        

    