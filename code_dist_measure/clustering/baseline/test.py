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
from sklearn.metrics.cluster import adjusted_rand_score as ar
from sklearn.metrics.cluster import rand_score as rs
from sklearn.metrics import normalized_mutual_info_score as nmi
import dtw
from fastdtw import fastdtw

def match_clustering(sequences, data, label, symbol_size):
    count = 0
    symbol_dist = sd.get_distance(symbol_size)
    pred_label = []
    for index in range(len(data)):
        # if label[index] == 1:
        #     continue
        dist = [sd.similar_match_dtw(symbol_dist, data[index], seq) for seq in sequences]
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
        pred_label.append(result)
        if result == label[index]:
            count += 1
        # else:
        #     print(data[index], label[index], result, dist)
    # print(count, len(data), len(label), count/len(data))
    return count/len(data), ar(label, pred_label), rs(label, pred_label)
#     # print(sequences)
#     # print()
#     dist = inter_distance( pred_label)
#     return dist

# def inter_distance( predict_label):
#     data = np.load('../../data/test_data.npy')
#     cluster = {i:[] for i in range(1, 7)}
#     for index in range(len(predict_label)):
#         cluster[predict_label[index]].append(index)
#     inter_dist = {i:0 for i in range(1, len(cluster)+1)}
#     for key in cluster:
#         count = 0
#         for i in range(len(cluster[key])):
#             for j in range(i+1, len(cluster[key])):
#                 count += 1
#                 inter_result, _ = fastdtw(data[cluster[key][i]], data[cluster[key][j]])
#                 inter_dist[key] += inter_result
#         if count > 0:
#             inter_dist[key] /= count
#     return inter_dist

# def read_data(file_path, label=False):
#     file = open(file_path, 'r')
#     data = []
#     while True:
#         line = file.readline()
#         if not line:
#             break
#         if label:
#             data.append(int(line))
#         else:
#             data.append(line.strip())
#     file.close()
#     return data
# def true_positive(clus, data, label):
#     symbol_dist = sd.get_distance(6)
#     TP = [0 for i in range(len(clus))]
#     for i in range(len(data)):
#         dist = [sd.similar_match_dtw(symbol_dist, data[i], seq) for seq in clus]
#         min_index = np.argmin(dist)
#         if label[i]-1 == min_index:
#             TP[min_index] += 1
#     return TP

# def match_clustering(result, centers):
#     symbol_dist = sd.get_distance(6)
#     have_transversed = [False for i in range(len(centers))]
#     dist_clus = [0 for i in range(len(centers))]
#     clus = []
#     for i in range(len(centers)):
#         dist = [sd.similar_match_dtw(symbol_dist, centers[i], seq) for seq in result]
#         min_index = 0
#         for j in range(len(have_transversed)):
#             if not have_transversed[i]:
#                 min_index = j
#                 break
#         min_dist = dist[min_index]
#         for j in range(min_index, len(have_transversed)):
#             if not have_transversed[j] and dist[j] < min_dist:
#                 min_index = j
#                 min_dist = dist[j]
#         dist_clus[i] = min_dist
#         have_transversed[min_index] = True
#         clus.append(result[min_index])
#     return dist_clus, clus
        

if __name__ == '__main__':
    path = '../../data/'
    sequences = [ 'cdabc', 'dabcd',  'abcdc']
    test_data = read_data(path+'/test_data.txt')
    test_label = read_data(path+'/test_label.txt', label=True)
    result = match_clustering(sequences, test_data, test_label, 6)
    print(result)