from tslearn.datasets  import  UCR_UEA_datasets
import numpy as np
import scipy.stats as stats
import sys
import math
import random 
import data_file_generate as dg
import sax_generate as sg
import symbol_distance as sd
from cdifflib import CSequenceMatcher 

def data_generate(size, length):
    tsd = UCR_UEA_datasets()
    data_train, label_train, data_test, label_test = tsd.load_dataset('Trace')
    data_train = data_train[label_train<4]
    label_train = label_train[label_train<4]
    data_test = data_test[label_test<4]
    label_test = label_test[label_test<4]
    data = np.vstack((data_train, data_test))
    label = np.hstack((label_train, label_test))
    data = data.reshape(data.shape[0], data.shape[1])
    symbol_size = size
    paa_length = length
    
    saxs = [sg.sax(elem, symbol_size, paa_length) for elem in data]
    sequences = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in saxs]  
    return sequences, label


def match_clustering(sequences, data, label, symbol_size):
    # print(sequences)
    # print(data[:10], label[:10])
    # print(data[50:60], label[50:60])
    # print(data[100:110], label[100:110])
    # print(len(data))
    count = 0
    symbol_dist = sd.get_distance(symbol_size)
    for index in range(len(data)):
        # if label[index] == 1:
        #     continue
        dist = [sd.similar_match(symbol_dist, data[index], seq) for seq in sequences]
        if len([lambda x: x==min(dist), dist]) > 1:
            min_dist = min(dist)
            max_s_value = 0
            can_index = 0
            for i in range(len(dist)):
                # if min_dist == dist[i] and data[index][0] == sequences[i][0]:
                if min_dist == dist[i]:
                    s_value = CSequenceMatcher(None, data[index], sequences[i]).ratio()
                    if max_s_value<s_value:
                        max_s_value = s_value
                        can_index = i
            result = can_index+1
        else:
            # print(data[index], label[index], dist)
            result = dist.index(min(dist))+1
        if result == label[index]:
            count += 1
        # else:
        #     print(data[index], label[index], result, dist)
    # print(count, len(data), len(label), count/len(data))
    return count/len(data)
    # print(sequences)
    # print()


if __name__ == '__main__':
    # sequences = [ 'cdabc', 'dabcd',  'abcdc']
    # data, label = data_generate(4, 10)
    # match_clustering(sequences, data, label)
    # s1 = CSequenceMatcher(None, ' abcd', 'bcde')
    # s2 = CSequenceMatcher(None, ' abcd', 'abc')
    # print(s1.ratio(), s2.ratio())
    window_length = 10
    symbol_size = 4
    import process as eg
    path = './data/Trace/'+str(symbol_size)+'_'+str(window_length)+'/'+str(80000)
    data = eg.read_data(path+'/data.txt')
    label = eg.read_data(path+'/label.txt', label=True)
    test_data = eg.read_data(path+'/test_data.txt')
    test_label = eg.read_data(path+'/test_label.txt', label=True)
    
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i in test_label:
        if i==1:
            count_1 += 1
        elif i==2:
            count_2 += 1
        else:
            count_3 += 1
    print(count_1, count_2, count_3)
    # print(test_label[:300])
    # data = []
    # label = []
    # for test_index in range(len(test_data)):
    #     if test_label[test_index]<4:
    #         data.append(test_data[test_index])
    #         label.append(test_label[test_index])
    # result = match_clustering(['ebeacd', 'ededcb', 'abaded', 'ebabab', 'abcbcd'], data, label, symbol_size)
    result = match_clustering(['cdabc', 'dcabc','abcd'], test_data, test_label, symbol_size)
    print(result)
