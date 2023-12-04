import numpy as np
import sax_generate as sg
import tree_growth as tg
import sys
# from cdifflib import CSequenceMatcher as distance_cal
import math
import symbol_distance as sd
import xxhash
import random
import multiprocessing as mp
import test
import os
import argparse
from sklearn.cluster import AgglomerativeClustering



def count_position_frequent_pattern_with_noise(data, label, epsilon, size, portion=0.1):
    choice = np.random.choice(len(data), int(len(data)*portion), replace=False)
    selected_data = [data[i] for i in choice]
    result = position_count_with_noise(selected_data, epsilon, size)
    data = [data[i] for i in range(len(data)) if i not in choice]
    label = [label[i] for i in range(len(label)) if i not in choice]
    return result, data, label

def OLH_mechanism(seed, elem, domain, epsilon):
    g = int(math.exp(epsilon)+1)
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1.0 / (math.exp(epsilon) + g - 1)
    result = (xxhash.xxh32(str(domain.index(elem)), seed=seed).intdigest() % g)
    p_sample = np.random.random_sample()
    if p_sample > p - q:
        result = np.random.randint(0, g)
    aggregation = []
    for i in domain:
        if xxhash.xxh32(str(domain.index(i)), seed=seed).intdigest() % g==result:
            aggregation.append(i)
    return aggregation

def GRR(elem, domain, epsilon):
    g = len(domain)
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    p_sample = np.random.random_sample()
    samples = [i for i in domain]
    if type(elem) != tuple:
        samples.remove(domain.index(elem))
    else:
        for i in samples:
            if i[0]==elem[0] and i[1]==elem[1]:
                samples.remove(i)
                break
    if p_sample>p:
        index =  random.choice(samples)
        return [index]
    return [elem]


def length_estimation(data, epsilon):
    # knwoledge of length: 3<=L<=15
    domain = [i for i in range(1, 15)]
    length_count = {}
    for i in range(len(data)):
        length = len(data[i])
        if length>14:
            length = 14
        # length_agg = OLH_mechanism(i, length, domain, epsilon)
        length_agg = GRR(length, domain, epsilon)
        for elem in length_agg:
            if elem not in length_count:
                length_count[elem] = 1
            else:
                length_count[elem] += 1
    # print(sorted(length_count.items(), key=lambda x: x[1], reverse=True)[0][0])
    result = sorted(length_count.items(), key=lambda x: x[1], reverse=True)
    if result[0][0] == 14:
        result = result[1][0]
    else:
        result = result[0][0]
    # print(length_count)
    # print(result)
    return result
    # return 5
    

def position_count_with_noise(data, epsilon, size):
    # length estimation
    length_ground_truth = {}
    for elem in data:
        if len(elem) not in length_ground_truth:
            length_ground_truth[len(elem)] = 1
        else:
            length_ground_truth[len(elem)] += 1
    # print(sorted(length_ground_truth.items(), key=lambda x: x[1], reverse=True)[0][0])
    
    # for _ in range(20):
    #     epsilon_length = epsilon/3
    #     length_epsilon_allocation = length_estimation(data, epsilon_length)
        
    #     choice = np.random.choice(len(data), int(len(data)*0.33), replace=False)
    #     selected_data = [data[i] for i in choice]
    #     length_population_allocation = length_estimation(selected_data, epsilon)
    #     print(length_epsilon_allocation, length_population_allocation)
    # sys.exit()
    choice = np.random.choice(len(data), int(len(data)*0.2), replace=False)
    selected_data = [data[i] for i in choice]
    data = [data[i] for i in range(len(data)) if i not in choice]
    # print("length estimation done!")
    length = length_estimation(selected_data, epsilon)
    
    # pattern candidate estimation
    candidates = sg.length_2_pattern(size)
    # print(candidates)
    count = {}
    for index in range(length):
        count[index] = {}
    # print(len(data))
    for elem in data:
        if len(elem)>length:
            elem = elem[:length]
        else:
            elem = elem+elem[-1]*(length-len(elem))
        index = np.random.randint(0, len(elem)-1)
        # elems_agg = OLH_mechanism(index, (elem[index], elem[index+1]), candidates, epsilon)
        elems_agg = GRR((elem[index], elem[index+1]), candidates, epsilon)
        for elem_ in elems_agg:
            if elem_ not in count[index]:
                count[index][elem_] = 1
            else:
                count[index][elem_] += 1
    for elem in count:
        count[elem] = sorted(count[elem].items(), key=lambda x: x[1], reverse=True)
    # print(count)
    return count


def select_data(data, label, num):
    choice = np.random.choice(len(data), num, replace=False)
    selected_data =  [data[i] for i in choice]
    data = [data[i] for i in range(len(data)) if i not in choice]
    label = [label[i] for i in range(len(label)) if i not in choice]
    return selected_data, data, label


def count_update_similar_match_with_noise(symbol_dist, sequence, root, level, epsilon, threshold=2):
    distance = {}
    sequences = {root:''}
    queue = [root]
    while queue:
        pointer = queue.pop(0)
        distance[pointer] = 0
        sequences[pointer] = sequences[pointer.parent] + pointer.value
        queue.extend(pointer.children)
    for node in sequences.keys():
        if node.level == level:
            distance[node] = sd.similar_match_dtw(symbol_dist, sequence[:len(sequences[node])], sequences[node])
        # print(distance[node], sequence, sequences[node])           
    # dist_nodes = list(filter(lambda x: x[0].level==level, sorted(distance.items(), key=lambda x: x[1])))
    
    # OUE mechanism
    # for node in dist_nodes:
    #     if distance[node[0]] <= threshold:
    #         node[0].count += np.random.binomial(1, p1)
    #     else:
    #         node[0].count += np.random.binomial(1, p0)
    # return root
    
    # RR mechanism
    # p = math.exp(epsilon) / (math.exp(epsilon) + len(dist_nodes) - 1)
    # q = 1.0 / (math.exp(epsilon) + len(dist_nodes) - 1)
    # p_sample = np.random.random_sample()
    # if p_sample>p:
    #     index = np.random.randint(1, len(dist_nodes))
    # else:
    #     index = 0
    # # print(index, len(dist_nodes))
    # dist_nodes[index][0].count += 1
    
    # distance match: exponential mechanism
    node_array = [node for node in sequences.keys() if node.level==level]
    can_dist = [1/(distance[node_array[i]]+0.1) for i in range(len(node_array))]
    # print(can_dist)
    if len(can_dist) == 0:
        can_score = [1/len(can_dist) for _ in range(len(can_dist))]
    elif max(can_dist) == min(can_dist):
        can_score = [1/len(can_dist) for _ in range(len(can_dist))]
    else:
        can_score = [np.exp(epsilon*(value-min(can_dist))/(max(can_dist)-min(can_dist))/2) for value in can_dist]
    can_score = can_score/np.linalg.norm(can_score, ord=1)
    # print(can_score)
    # sys.exit()
    can_index = np.random.choice(len(can_score), p=can_score)
    node_array[can_index].count += 1
    
    # # similar sequence adding
    # seq_can = {}
    # for node in node_array:
    #     pointer = node
    #     string = ''
    #     while pointer != root:
    #         string += pointer.value
    #         pointer = pointer.parent
    #     seq_can[node] = string[::-1]
    # for node in node_array:
    #     if sd.similar_match(symbol_dist, seq_can[node_array[can_index]], seq_can[node]) <= threshold:
    #         node.count += 1
    return root
    

        
def generate_tree(symbol_dist, knowledge, data, label, epsilon, threshold):
    # print(knowledge)
    # clustering number
    k = 6
    knowledge_threshold = 50
    
    data_length = len(data)
    root = tg.Node(0, '', -1)
    root.parent = root
    
    levels_node = {-1:[root]}
    max_level = max(knowledge.keys())
    
    for level in range(max_level+1):
        # print("level", level)
        levels_node[level] = []
        if level == 0:
            candidates = knowledge[level][:3*k]
            # print(candidates)
            for can_index in range(len(candidates)):
                elem = candidates[can_index][0]
                count = candidates[can_index][1]
                if True:
                    has_add = False
                    for node in levels_node[level]:
                        if node.value == elem[0]:
                            has_add = True
                            break
                    if not has_add:
                        node1 = tg.Node(0, elem[0], level)
                        node1.parent = root
                        root.children.append(node1)
                        levels_node[level].append(node1)
            selected_data, data, label = select_data(data, label, math.floor(data_length*0.7/max_level))
            # print_tree(root)
            for seq in selected_data:
                root = count_update_similar_match_with_noise(symbol_dist, seq, root, level, epsilon)
            
            # prune
            # for node in levels_node[level]:
            #     if node.count < threshold:
            #         node.prune()
            # print_tree(root)
            sorted_nodes = sorted(levels_node[level], key=lambda x: x.count, reverse=True)
            for node_index in range(len(sorted_nodes)):
                if node_index >= 3*k:
                    sorted_nodes[node_index].prune()
                #     sorted_nodes[node_index].count /= len(selected_data)
                # else:
                #     sorted_nodes[node_index].count /= len(selected_data)
            # sys.exit()
        else:
            candidates = knowledge[level-1][:3*k]
            for previous_node in levels_node[level-1]:
                if previous_node.is_pruned:
                    continue
                for elem, count in candidates:
                    # if count>knowledge_threshold and elem[0] == previous_node.value:
                    if elem[0] == previous_node.value:
                        node1 = tg.Node(0, elem[1], level)
                        node1.parent = previous_node
                        previous_node.children.append(node1)
                        levels_node[level].append(node1)
            selected_data, data, label = select_data(data, label, math.floor(data_length*0.7/max_level))
            # print(len(data), len(label))
            for seq in selected_data:
                root = count_update_similar_match_with_noise(symbol_dist, seq, root, level, epsilon)
                
            # node prune
            # for node in levels_node[level]:
            #     if node.count < threshold:
            #         node.prune()
            sorted_nodes = sorted(levels_node[level], key=lambda x: x.count, reverse=True)
            for node_index in range(len(sorted_nodes)):
                if node_index >= 3*k:
                    sorted_nodes[node_index].prune()
                #     sorted_nodes[node_index].count /= len(selected_data)
                # else:
                #     sorted_nodes[node_index].count /= len(selected_data)
            # print_tree(root)
             
        # print("population", len(data))
    return root, levels_node, data, label


def print_tree(root):
    sequences = {root:''}
    queue = [root]
    while queue:
        pointer = queue.pop(0)
        sequences[pointer] = sequences[pointer.parent] + pointer.value
        queue.extend(pointer.children)
    levels = {}
    for node in sequences.keys():
        if node.level not in levels:
            levels[node.level] = [node]
        else:
            levels[node.level].append(node)
    for level in levels.keys():
        print('Level', level)
        for node in levels[level]:
            print(sequences[node], node.count)
        print()
 
 
# utilize GRR to match label
def label_match_with_noise(root, symbol_size, levels, data, epsilon, clus_num):
    # print("clus_num", clus_num)
    symbol_dist = sd.get_distance(symbol_size)
    # print(len(data), len(label))
    # data = data[:30]
    nodes = []
    for node in levels[max(levels.keys())]:
        pointer = node
        if pointer.is_pruned:
            continue
        string = ''
        count = node.count
        while pointer != root:
            string += pointer.value
            pointer = pointer.parent
        nodes.append((string[::-1], count))
    candidates = [elem[0] for elem in nodes]
    # print(len(candidates))
    # print(candidates)
    
    distance_matrix = [[0 for i in range(len(candidates))] for j in range(len(candidates))]
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            dist = sd.similar_match_dtw(symbol_dist, candidates[i], candidates[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    clus = AgglomerativeClustering(n_clusters=clus_num, metric='precomputed', linkage='average')
    clus.fit(distance_matrix)
    y_pred = clus.labels_
    centers = {}
    for i in range(clus_num):
        centers[i] = {}
        for j in range(len(y_pred)):
            if i==y_pred[j]:
                centers[i][candidates[j]] = 0
    
    for d in data:
        min_dist = sys.maxsize
        min_index = 0
        for i in range(len(candidates)):
            dist = sd.similar_match_dtw(symbol_dist, candidates[i], d)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        min_index = GRR(min_index, range(len(candidates)), epsilon)[0]
        centers[y_pred[min_index]][candidates[min_index]] += 1
    # print(centers)
    
    result = []
    for i in range(clus_num):
        result.append(sorted(centers[i].items(), key=lambda x: x[1], reverse=True)[0][0])
    # print(result)
    # sys.exit()
    return result


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

def process_file(train_data, train_label, times, symbol_size, epsilon, core_index, symbol_distance, test_data, test_label):
    sum_acc = 0
    # c_f = open('./result_centroids/'+str(epsilon)+'/'+str(core_index)+'_centroids.txt', 'w+')
    sum_ari = 0
    sum_rs = 0
    t_count = 0
    while t_count<times:
        try:
            data = train_data[:]
            label = train_label[:]
            # print("data generate done!")
            knowledge_noised, data, label = count_position_frequent_pattern_with_noise(data, label, epsilon, symbol_size)
            # knowledge, _, _ = count_position_frequent_pattern(data, label)
            # print(knowledge_noised)
            # print()
            # print(knowledge)
            # print("knowledge done!")
            if epsilon == 1:
                threshold = 200
            if epsilon == 2:
                threshold = 300
            else:
                threshold = 500
            root, levels, data, label = generate_tree(symbol_distance, knowledge_noised, data, label, epsilon, threshold)
            # print_tree(root)
            # print(knowledge_noised)
            # print(levels[4])
            # print("tree generate done!")
            result = label_match_with_noise(root, symbol_size, levels, data, epsilon, clus_num=6)
            # c_f.write(str(result)+'\n')
            # print("label match done!")
            acc, ari, ri = test.match_clustering(result, test_data, test_label, symbol_size)
            
            # print(result, acc, ari, epsilon, t_count, times)
            # sys.exit()
            sum_acc += acc
            sum_ari += ari
            sum_rs += ri
            # print("test done!")
            t_count += 1
        except:
            continue
    file = open('./result/'+str(epsilon)+'/acc_'+str(core_index+30)+'.txt', 'w')
    file.write(str(sum_ari)+'\n')
    file.close()
    # print("average accuracy:", sum_acc/times)
    # print("adjusted rand index:", sum_ari/times)
    # print("rand index", sum_rs/times)
    
    

if __name__ == '__main__':
    symbol_size = 6
    window_length = 25
    core_number = 28
    
    
    symbol_distance = sd.get_distance(symbol_size)
    
    epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    # epsilons = [0.5]
    # epsilons = [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12.5, 13, 13.5, 14.5, 15, 15.5, 16]
    # epsilons = [4]
    # epsilons = [8.5, 9, 9.5, 10, 10.5, 11, 11.5 ]
    # value = sys.argv[1]
    # epsilons = [float(value)]
    
    result = []
    for epsilon in epsilons:
        print("epsilon =", epsilon)
        # folder = os.path.exists('./result/'+str(epsilon))
        # if not folder:
        #     os.makedirs('./result/'+str(epsilon))
        # else:
        #     files = os.listdir('./result/'+str(epsilon))
        #     for path in files:
        #         os.remove('./result/'+str(epsilon)+'/'+path)
                
        # folder = os.path.exists('./result_centroids/'+str(epsilon))
        # if not folder:
        #     os.makedirs('./result_centroids/'+str(epsilon))
        # else:
        #     files = os.listdir('./result_centroids/'+str(epsilon))
        #     for path in files:
        #         os.remove('./result_centroids/'+str(epsilon)+'/'+path)
        
        # path = '../data/'
        # data = read_data(path+'/data.txt')
        # label = read_data(path+'/label.txt', label=True)
        # test_data = read_data(path+'/test_data.txt')
        # test_label = read_data(path+'/test_label.txt', label=True)
        # print("initial", len(data), len(label))
        
        # # print("test data generate done!")
        sum_accuracy = 0
        times = 18
        # for _ in range(times):
        # process_file(data, label, 1, symbol_size, epsilon, -1, symbol_distance, test_data, test_label)
        # pool = mp.Pool(core_number)
        # for i in range(core_number):
        #     pool.apply_async(process_file, args=(data, label, times, symbol_size, epsilon, i, symbol_distance, test_data, test_label,))
        # pool.close()
        # pool.join()
        
        
        # for i in range(core_number):
        #     file = open('./result/'+str(epsilon)+'/acc_'+str(i)+'.txt', 'r')
        #     sum_accuracy += float(file.readline())
        #     file.close()
        # print("average accuracy:", sum_accuracy/core_number/times)
        
        
        files = os.listdir('./result/'+str(epsilon))
        for path in files:
            file = open('./result/'+str(epsilon)+'/'+path, 'r')
            sum_accuracy += float(file.readline())
            file.close()
        print("average accuracy:", epsilon, sum_accuracy/len(files)/times)
        result.append(sum_accuracy/len(files)/times)
    file = open('result.npy', 'wb')
    np.save(file, result)