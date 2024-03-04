import numpy as np 
from Hybrid  import *
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from multiprocessing import Pool
import os


# path = '/data/fat/maoyl/ShapeExtraction/'+'07_09'+'/patternLDP/'
path = '/data/fat/maoyl/ShapeExtraction/'+'without_sax'+'/patternLDP/'

def class_rate(para):
    epsilon, index, train_data, train_label = para
    np.random.seed(index)
    # train_data = np.load(path+str(epsilon)+'/'+str(index)+'_data.npy')
    # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
    # train_label = np.load(path+str(epsilon)+'/'+str(index)+'_label.npy')
    clf = RandomForestClassifier(min_samples_leaf=1)
    clf.fit(train_data, train_label)
    
    test_data = np.load('./data/test_data.npy')
    test_label = np.load('./data/test_label.npy')
    acc = clf.score(test_data, test_label)
    # print(acc)
    return acc


def process(para):
    epsilon, index = para[0], para[1]
    np.random.seed(index)
    train_data = np.load('./data/train_data.npy')
    train_label = np.load('./data/train_label.npy')
    train_data = 2*(train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))-1
    
    # perturb label
    perturbed_data = []
    perturbed_label = []
    p = np.exp(epsilon/3)/(np.exp(epsilon/3)+2)
    q = 1/(np.exp(epsilon/3)+2)
    for i in range(len(train_label)):
        # if i>10:
        #     exit()
        d = hybrid(train_data[i], -1, 1, epsilon/3*2/len(train_data[i]))
        new = [d[i] for i in range(len(d))]
        perturbed_data.append(new)
        # print(perturbed[0])
        probs = [q, q, q]
        probs[train_label[i]-1] = p
        sample = np.random.choice([1, 2, 3], p=probs)
        perturbed_label.append(sample)
    train_data = np.array(perturbed_data)
    train_label = np.array(perturbed_label)
    acc = class_rate((epsilon, index, train_data, train_label))
    write_result((epsilon, acc))
    return 


def write_result(para):
    epsilon, acc = para[0], para[1]
    file = open('./result_12_04/'+str(epsilon)+"_result.txt", "a+")
    file.write(str(acc)+"\n")
    file.close()


if __name__ == '__main__':
    epsilons = [8]
    # epsilons = [7.5, 8]
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
        for index in range(470, 500):
            para.append((epsilon, index))
    # process(para[0])
    # for i in range(10):
    # class_rate(para[0])
    core_num = 28
    pool = Pool(core_num)
    for i in range(len(para)):
            pool.apply_async(process, args=(para[i],), callback=write_result)
    pool.close()
    pool.join()
    
    
    