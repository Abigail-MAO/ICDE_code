from pyts.metrics import dtw
import os
import numpy as np
from multiprocessing import Pool


def read_data(file_name):
    f = open(file_name, 'r')
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        line = list(map(lambda x: float(x[1:len(x)-1]), line.strip().split(',')))
        data.append(line)
    f.close()
    return data

def distance_cal(para):
    index, epsilon = para[0], para[1]
    path = path = '/data/fat/maoyl/ShapeExtraction/'+'05_30/'+'patternLDP_result/'+str(epsilon)+'/'+str(index)+'_perturbed.txt'
    data = read_data(path)
    # print(len(data))
    # print([len(data[i]) for i in range(5)])
    dist = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            # print(i, j)
            dist[i][j] = dtw(data[i], data[j])
            dist[j][i] = dist[i][j]
    file = open('../result/'+'patternLDP/dist/'+str(epsilon)+'/'+str(index)+'_dist.npy', 'wb')
    np.save(file, dist)
    file.close()
    pass 


if __name__ == '__main__':
    epsilons = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    # epsilons = [0.5]
    para = []
    for epsilon in epsilons:
        print("epsilon =", epsilon)
        folder = os.path.exists('../result/'+'patternLDP/dist/'+str(epsilon))
        if not folder:
            os.makedirs('../result/'+'patternLDP/dist/'+str(epsilon))
        else:
            files = os.listdir('../result/'+'patternLDP/dist/'+str(epsilon))
            for path in files:
                os.remove('../result/'+'patternLDP/dist/'+str(epsilon)+'/'+path)
        for index in range(500):
            para.append([index, epsilon])
            
    # distance_cal(para[0])
    core_num = 28
    pool = Pool(core_num)
    for i in range(len(para)):
        pool.apply_async(distance_cal, (para[i],))
    pool.close()
    pool.join()
    
            
            
            
            
                