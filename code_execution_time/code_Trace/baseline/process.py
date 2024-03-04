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
import time


def count_position_frequent_pattern_with_noise(data, label, epsilon, size, portion=0.1):
    choice = np.random.choice(len(data), int(len(data)*portion), replace=False)
    selected_data = [data[i] for i in choice]
    result, pro_time = position_count_with_noise(selected_data, epsilon, size)
    data = [data[i] for i in range(len(data)) if i not in choice]
    label = [label[i] for i in range(len(label)) if i not in choice]
    return result, data, label, pro_time

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
    q = 1.0 / (math.exp(epsilon) + g - 1)
    p_sample = np.random.random_sample()
    samples = [i for i in range(len(domain))]
    samples.remove(domain.index(elem))
    if p_sample>p:
        index =  random.choice(samples)
        return [domain[index]]
    return [elem]


def length_estimation(data, epsilon):
    # knwoledge of length: 3<=L<=15
    domain = [i for i in range(1, 11)]
    length_count = {}
    for i in range(len(data)):
        length = len(data[i])
        if length>10:
            length = 10
        length_agg = OLH_mechanism(i, length, domain, epsilon)
        for elem in length_agg:
            if elem not in length_count:
                length_count[elem] = 1
            else:
                length_count[elem] += 1
    # print(sorted(length_count.items(), key=lambda x: x[1], reverse=True)[0][0])
    result = sorted(length_count.items(), key=lambda x: x[1], reverse=True)[0][0]
    return result
    # return 5
    

def position_count_with_noise(data, epsilon, size):
    # length estimation
    length_time = time.time()
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
    length_time = time.time()-length_time
    estimation_time = time.time()
    length = length_estimation(selected_data, epsilon)
    estimation_time = time.time()-estimation_time
    
    # pattern candidate estimation
    candidates = sg.length_2_pattern(size)
    count = {}
    for index in range(length):
        count[index] = {}
    for elem in data:
        if len(elem)>length:
            elem = elem[:length]
        # else:
        #     elem = elem+'a'*(length-len(elem))
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
    return count, estimation_time/len(selected_data)+length_time


def select_data(data, label, num):
    choice = np.random.choice(len(data), num, replace=False)
    selected_data =  [data[i] for i in choice]
    data = [data[i] for i in range(len(data)) if i not in choice]
    label = [label[i] for i in range(len(label)) if i not in choice]
    return selected_data, data, label

def select_data_with_label(data, label, num):
    choice = np.random.choice(len(data), num, replace=False)
    selected_data =  [data[i] for i in choice]
    selected_label = [label[i] for i in choice]
    data = [data[i] for i in range(len(data)) if i not in choice]
    label = [label[i] for i in range(len(label)) if i not in choice]
    return selected_data, selected_label, data, label


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
            distance[node] = sd.similar_match(symbol_dist, sequence[:len(sequences[node])], sequences[node])
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
    if max(can_dist) == min(can_dist):
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
    
def count_update_similar_match_with_noise_label(symbol_dist, sequence, root, level, epsilon, label, label_count, threshold=2):
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
            distance[node] = sd.similar_match(symbol_dist, sequence[:len(sequences[node])], sequences[node])
    
    # distance match: exponential mechanism
    node_array = [node for node in sequences.keys() if node.level==level]
    can_dist = [1/(distance[node_array[i]]+0.1) for i in range(len(node_array))]
    if max(can_dist) == min(can_dist):
        can_score = [1/len(can_dist) for _ in range(len(can_dist))]
    else:
        can_score = [np.exp(epsilon/3*2*(value-min(can_dist))/(max(can_dist)-min(can_dist))/2) for value in can_dist]
    can_score = can_score/np.linalg.norm(can_score, ord=1)
    # print(can_score)
    # sys.exit()
    can_index = np.random.choice(len(can_score), p=can_score)
    node_array[can_index].count += 1
    
    # label perturb
    p = np.exp(epsilon/3)/(np.exp(epsilon/3)+2)
    q = 1/(np.exp(epsilon/3)+2)
    probs = [q, q, q]
    probs[label-1] = p
    l = np.random.choice([1, 2, 3], p=probs)
    label_count[node_array[can_index]][l] += l
    return root, label_count
        
        
        
def generate_tree(symbol_dist, knowledge, data, label, epsilon, threshold):
    # print(knowledge)
    # clustering number
    k = 3
    knowledge_threshold = 50
    
    pro_time = time.time()
    level2_time = 0
    level2_users = 0

    
    data_length = len(data)
    root = tg.Node(0, '', -1)
    root.parent = root
    
    levels_node = {-1:[root]}
    max_level = max(knowledge.keys())
    
    # print(max_level)
    for level in range(max_level+1):
        levels_node[level] = []
        if level == 0:
            candidates = knowledge[level][:]
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
            selected_data, data, label = select_data(data, label, int(data_length/(max_level+1)))
            # print_tree(root)
            level1_time = time.time()
            for seq in selected_data:
                root = count_update_similar_match_with_noise(symbol_dist, seq, root, level, epsilon)
            level1_time = time.time()-level1_time
            level1_users = len(selected_data)
            # prune
            for node in levels_node[level]:
                if node.count < threshold:
                    node.prune()
            # print_tree(root)
            # sorted_nodes = sorted(levels_node[level], key=lambda x: x.count, reverse=True)
            # for node_index in range(len(sorted_nodes)):
            #     if node_index >= 2*k:
            #         sorted_nodes[node_index].prune()
            #         sorted_nodes[node_index].count /= len(selected_data)
            #     else:
            #         sorted_nodes[node_index].count /= len(selected_data)
            # sys.exit()
        else:
            candidates = knowledge[level-1][:]
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
           
            # print(len(selected_data))
            if level == max_level:
                # print(len(data))
                label_count = {}
                for node in levels_node[level]:
                    label_count[node] = {1:0, 2:0, 3:0}
                level3_time = time.time()
                for seq_i in range(len(data)):
                    root, label_count = count_update_similar_match_with_noise_label(symbol_dist, data[seq_i], root, level, epsilon, label[seq_i], label_count)
                level3_time = time.time()-level3_time
                level3_users = len(data)
                
            else:
                selected_data, data, label = select_data(data, label, int(data_length/(max_level+1)))
                # print(len(selected_data))
                other_time = time.time()
                for seq in selected_data:
                    root = count_update_similar_match_with_noise(symbol_dist, seq, root, level, epsilon)
                other_time = time.time()-other_time
                level2_time  += other_time
                level2_users += len(selected_data)
                
            # node prune
            # remain_time = time.time()
            node_reminded = len(levels_node[level])
            sorted_nodes = sorted(levels_node[level], key=lambda x: x.count, reverse=True)
            for node_index in range(len(sorted_nodes)):
                if sorted_nodes[node_index].count < threshold:
                    sorted_nodes[node_index].prune()
                    node_reminded -= 1
                # if node_reminded <= 5*k:
                #     break
            # print_tree(root)
            # sorted_nodes = sorted(levels_node[level], key=lambda x: x.count, reverse=True)
            # for node_index in range(len(sorted_nodes)):
            #     if node_index >= 2*k:
            #         sorted_nodes[node_index].prune()
            #         sorted_nodes[node_index].count /= len(selected_data)
            #     else:
            #         sorted_nodes[node_index].count /= len(selected_data)
        
        # print("population", len(data))
        # remain_time = time.time()-remain_time
    pro_time = time.time()-pro_time
    print("tree ge", pro_time-level1_time-level2_time-level3_time, level1_time, level2_time, level3_time)
    pro_time = pro_time-level1_time-level2_time-level3_time+level1_time/level1_users+level2_time/level2_users+level3_time/level3_users
    
    return root, levels_node, data, label, label_count, pro_time



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
def label_match_with_noise(root, symbol_size, levels, data, label, epsilon, label_count, clus_num):
    pre_time = time.time()
    symbol_dist = sd.get_distance(symbol_size)
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
        nodes.append((string[::-1], label_count[node]))
    centroids = []
    for i in range(clus_num):
        class_ = i+1 
        max_count = 0
        for node in nodes:
            if node[1][class_]>max_count and node[0] not in centroids:
                max_count = node[1][class_]
                max_node = node
        centroids.append(max_node[0])
    # print(centroids)
    return centroids, time.time()-pre_time


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
    np.random.seed(times)
    sum_acc = 0
    # c_f = open('./result_centroids/'+str(epsilon)+'/'+str(core_index)+'_centroids.txt', 'w+')
    c_count = 0
    consume_time = 0
    
    data = train_data[:]
    label = train_label[:]
    # print("data generate done!")
    knowledge_noised, data, label, time_pattern = count_position_frequent_pattern_with_noise(data, label, epsilon, symbol_size)
    # knowledge, _, _ = count_position_frequent_pattern(data, label)
    # print(knowledge_noised)
    # print()
    # print(knowledge)
    # print("knowledge done!")
    if epsilon == 0.5:
        threshold = 100
    elif epsilon == 1:
        threshold = 100
    elif epsilon == 2:
        threshold = 100
    else:
        threshold = 100
    root, levels, data, label, label_count, time_gt = generate_tree(symbol_distance, knowledge_noised, data, label, epsilon, threshold)
    # print_tree(root)
    # sys.exit()
    # print(knowledge_noised)
    # print(levels[4])
    # print("tree generate done!")
    result, time_matching = label_match_with_noise(root, symbol_size, levels, data, label, epsilon, label_count, clus_num=3)
    # c_f.write(str(result)+'\n')
    # print("label match done!")
    acc, testint_time = test.match_clustering(result, test_data, test_label, symbol_size)
    # print(result, acc)
    sum_acc += acc
    # print(sum_acc)
    # print("test done!")
    c_count += 1
    print(time_pattern, time_gt, time_matching, testint_time)
    return time_pattern+time_gt+time_matching+testint_time

    
    # while c_count<times:
    #     try:
    #         start_time = time.time()
    #         data = train_data[:]
    #         label = train_label[:]
    #         # print("data generate done!")
    #         knowledge_noised, data, label, time_pattern = count_position_frequent_pattern_with_noise(data, label, epsilon, symbol_size)
    #         # knowledge, _, _ = count_position_frequent_pattern(data, label)
    #         # print(knowledge_noised)
    #         # print()
    #         # print(knowledge)
    #         # print("knowledge done!")
    #         if epsilon == 0.5:
    #             threshold = 100
    #         elif epsilon == 1:
    #             threshold = 100
    #         elif epsilon == 2:
    #             threshold = 100
    #         else:
    #             threshold = 100
    #         root, levels, data, label, label_count, time_gt = generate_tree(symbol_distance, knowledge_noised, data, label, epsilon, threshold)
    #         # print_tree(root)
    #         # sys.exit()
    #         # print(knowledge_noised)
    #         # print(levels[4])
    #         # print("tree generate done!")
    #         result, time_matching = label_match_with_noise(root, symbol_size, levels, data, label, epsilon, label_count, clus_num=3)
    #         # c_f.write(str(result)+'\n')
    #         # print("label match done!")
    #         acc, testint_time = test.match_clustering(result, test_data, test_label, symbol_size)
    #         # print(result, acc)
    #         sum_acc += acc
    #         # print(sum_acc)
    #         # print("test done!")
    #         c_count += 1
    #         consume_time += time.time()-start_time
    #     except:
    #         continue
    # file = open('./time/'+str(epsilon)+'/acc_'+str(core_index)+'.txt', 'w')
    # file.write(str(consume_time)+'\n')
    # file.close()
    
    

if __name__ == '__main__':
    symbol_size = 4
    window_length = 10
    core_number = 28
    
    
    symbol_distance = sd.get_distance(symbol_size)
    
    epsilons = [4]
    # epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

    acc_result = []
    for epsilon in epsilons:
        print("epsilon =", epsilon)
        folder = os.path.exists('./time/'+str(epsilon))
        if not folder:
            os.makedirs('./time/'+str(epsilon))
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
        
        
        
        data_amount = 40000
        path = '../data/'
        data = read_data(path+'/data.txt')
        label = read_data(path+'/label.txt', label=True)
        test_data = read_data(path+'/test_data.txt')
        test_label = read_data(path+'/test_label.txt', label=True)
        
        
        # # print("test data generate done!")
        sum_accuracy = 0
        times = 5
        sum_time = 0
        for _ in range(times):
            pro_time = process_file(data, label, _, symbol_size, epsilon, 0, symbol_distance, test_data, test_label)
            sum_time += pro_time
            print(_+1, sum_time, sum_time/(_+1))
        print(sum_time/times)
        # pool = mp.Pool(core_number)
        # for i in range(core_number):
        #     pool.apply_async(process_file, args=(data, label, times, symbol_size, epsilon, i, symbol_distance, test_data, test_label,))
        # pool.close()
        # pool.join()
        
        
    #     for i in range(core_number):
    #         file = open('./result/'+str(epsilon)+'/acc_'+str(i)+'.txt', 'r')
    #         sum_accuracy += float(file.readline())
    #         file.close()
    #     print("average accuracy:", sum_accuracy/core_number/times)
    #     acc_result.append(sum_accuracy/core_number/times)
    # print(acc_result)

    #     files = os.listdir('./time/'+str(epsilon))
    #     for path in files:
    #         file = open('./time/'+str(epsilon)+'/'+path, 'r')
    #         sum_accuracy += float(file.readline())
    #         file.close()
    #     print("average accuracy:", sum_accuracy/len(files)/times)
    #     acc_result.append(sum_accuracy/core_number/times)
    # print(acc_result)

    # 59.62680198843517