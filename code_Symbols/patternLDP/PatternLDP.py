import read_file as rf
import pattern_aware_sampling as pas
import importance_characterization as ic
import importance_aware_randomization as iar
import dynamic_time_wrapping as dt
from matplotlib import pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
import math
import sys

delta = 0.5
theta = 10
mu = 1
k_p = 0.8
k_i = 0.1
k_d = 0.5
pi = 5

fig_index = 0

def pattern_ldp(data, delta, theta, mu, epsilon, k_p, k_i, k_d, pi):
        time = [index for index in range(len(data)+2)]
        sample_points = pas.sample_points(data, time, delta)
        
        # global fig_index
        # plt.plot(data)
        # plt.scatter(sample_points, [data[i] for i in sample_points], color='r')
        # plt.savefig('./sample_points/'+str(fig_index)+'.png')
        # plt.cla()
        # fig_index += 1
        sample_points.insert(0, 1)
        sample_points.insert(0, 0)
        # print(len(sample_points), sample_points)
        
        perturb_array = []
        error_array = [0, 0]
        score_array = [1, 1]
        data = np.insert(data, 0, data[0])
        data = np.insert(data, 0, data[0])
        
        alpha = 0.5
        epsilon = epsilon/3*2
        epsilon_remaind = epsilon
        w = len(data)
        
        range_array = []
        epsilon_allocation = []
        for elem_index in range(2, len(sample_points)):
            # if index <= 1:
            #     perturb_array.append(data[sample_points[index]])
            # if index>1:
                score_array, error_array = ic.pid_control(elem_index, data, time, error_array, sample_points, k_p, k_i, k_d, pi, score_array)
        score_array = score_array[2:]
        sorted_score = sorted(score_array)
        sorted_index = np.argsort(score_array)
        for elem_index in range(1, len(sorted_score)):
            if sorted_score[elem_index]>sorted_score[elem_index-1]+3:
                sorted_score[elem_index] = sorted_score[elem_index-1]+3
        for elem_index in range(len(sorted_index)):
            score_array[sorted_index[elem_index]] = sorted_score[elem_index]
        # print(sorted_score, sorted_index)
        
        
        b = max([math.log(theta/score_array[i]+mu) for i in range(len(score_array))])
        # score_array[0] = 1
        # print("score_array")
        # for elem_index in range(len(sample_points)-2):
        #         epsilon_remaind, epsilon_now, b = iar.range(elem_index, alpha, theta, mu, score_array, epsilon_remaind, epsilon, w)
        #         epsilon_allocation.append(epsilon_now)
        #         range_array.append(b)
        # b = max(range_array)
        # print("epsilon", epsilon_allocation)
        # epsilon_allocation = [epsilon/(len(sample_points)-2) for i in range(len(sample_points)-2)]
        sum_score = sum(score_array)
        epsilon_allocation = [epsilon*score_array[i]/sum_score for i in range(len(score_array))]
        # print("epsilon", epsilon_allocation)
        

        for index in range(len(sample_points)-2):
            perturb_result = iar.perturb(data, index, b, epsilon_allocation[index], sample_points)
            perturb_array.append(perturb_result)
            # if index % w == 0:
            #     print(index, "here")
            #     alpha = 0.5
            #     epsilon_remaind = epsilon
                # if index % 100 == 0:
                #     print("100 done")
        # print("perturb_array")
        # plt.scatter(sample_points, perturb_array)
        
        sample_points = sample_points[2:]
        process_array = []
        start_index = sample_points[0]
        for index in range(1, len(sample_points)):
            process_array.append(perturb_array[index-1])
            end_index = sample_points[index]
            k = (perturb_array[index]-perturb_array[index-1])/(end_index-start_index)
            for virtual_index in range(start_index+1, end_index):
                # process_array.append(k*(time[virtual_index]-time[start_index])+data[start_index])
                es_v = k*(virtual_index-start_index)+perturb_array[index-1]
                # print(es_v)
                process_array.append(es_v)
            start_index = end_index
        # plt.plot(data)
        # plt.scatter(sample_points, [data[i] for i in sample_points])
        # plt.plot(process_array)
        # plt.ylim(-10, 10)
        # plt.savefig('test1.png')
        # sys.exit(0)
        # print("process_array")
        return process_array

def read_data(file_path, n):
    all_amount = 22013
    # np.random.seed(2023)
    file = open(file_path, 'r')
    data = []
    indices = []
    index = 0
    while True:
        line = file.readline()
        if not line:
            break
        line = list(map(lambda x: float(x), line.strip().split(',')))
        if np.random.binomial(1, n/all_amount) == 1:
            indices.append(index)
            data.append(line)
        if len(data)==n:
            break
        index += 1
    file.close()
    # print(indices)
    return data


def perturb(para):
        data, label, index, epsilon = para[0], para[1], para[2], para[3]
        perturbed_data = []
        perturbed_label = []
        
        p = np.exp(epsilon/3)/(np.exp(epsilon/3)+2)
        q = 1/(np.exp(epsilon/3)+2)
        for i in range(len(data)):
            # if i>10:
            #     exit()
            d = pattern_ldp(data[i], delta, theta, mu, epsilon, k_p, k_i, k_d, pi)
            new = [d[i] for i in range(len(d))]
            perturbed_data.append(new)
            # print(perturbed[0])
            # probs = [q, q, q]
            # probs[label[i]-1] = p
            # sample = np.random.choice([1, 2, 3], p=probs)
            # perturbed_label.append(sample)
            # print("perturbed data")
        # colors = ['r', 'g', 'b']
        # for i in range(20):
        #     # if label[i]==1:
        #     # plt.plot(data[i])
        #     plt.plot(perturbed_data[i], color=colors[label[i]-1])
        #     plt.ylim(-10, 10)
        #     print(perturbed_label[i], label[i])
        # plt.savefig('test1.png')
        # plt.cla()
        # sys.exit(0)
        
        
        # print(len(perturbed_data), len(perturbed_data[0]))
        # file = open(path+'/patternLDP/'+str(epsilon)+'/'+str(index)+'_perturbed.txt', 'w')
        # for i in range(len(perturbed_data)):
        #     for j in range(len(perturbed_data[i])-1):
        #         file.write(str(perturbed_data[i][j])+',')
        #     file.write(str(perturbed_data[i][-1]))
        #     file.write('\n')
        # file.close()
        perturbed_data = np.array(perturbed_data)  
        # print("here")
        file = open(path+'/patternLDP/'+str(epsilon)+'/'+str(index)+'_data.npy', 'wb')
        np.save(file, perturbed_data)
        file.close()
        perturbed_label = np.array(perturbed_label)
        file = open(path+'/patternLDP/'+str(epsilon)+'/'+str(index)+'_label.npy', 'wb')
        np.save(file, perturbed_label)
        file.close()
        
        
        
if __name__ == "__main__":
    
    symbol_size = 4
    window_length = 10
    
    # data = np.load('../data/train_data.npy')
    # label = np.load('../data/train_label.npy')
    # path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/train/'
    data = np.load('../data/test_data.npy')
    label = np.load('../data/test_label.npy')
    path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/test/'
    
    core_num = 30
    pool = Pool(core_num)
    
    epsilons = [7.5, 8]
    # epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    para = []
    for epsilon in epsilons:
        print("epsilon =", epsilon)
        folder = os.path.exists(path+'/patternLDP/'+str(epsilon))
        if not folder:
            os.makedirs(path+'/patternLDP/'+str(epsilon))
        # else:
        #     files = os.listdir(path+'/patternLDP/'+str(epsilon))
        #     for f_path in files:
        #         os.remove(path+'/patternLDP/'+str(epsilon)+'/'+f_path)
        for index in range(100):
            para.append((data, label, index, epsilon))
    
    # for i in range(10):
    #     perturb(para[i])
    for i in range(len(para)):
            pool.apply_async(perturb, args=(para[i],))
    pool.close()
    pool.join()
    exit()
            
    
      
        
        
        
        
        