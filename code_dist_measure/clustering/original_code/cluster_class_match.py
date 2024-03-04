import numpy as  np
from collections import Counter

def most_frequent(arr):
    counter = Counter(arr)
    most_common = counter.most_common(1)
    return most_common[0][0]

def match_clustering(pred_label, label):
    clusters = {i:[] for i in range(1, 7)}
    for clus in range(1, 7):
        for index in range(len(pred_label)):
            if pred_label[index] == clus-1:
                clusters[clus].append(label[index])
    flag = True
    for clus in clusters:
        if len(clusters[clus]) == 0:
            flag = False
            return flag
    clus_label = []
    for clus in clusters:
        clus_label.append(most_frequent(clusters[clus]))
    return clus_label
        