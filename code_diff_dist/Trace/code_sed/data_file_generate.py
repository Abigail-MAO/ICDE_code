import sax_generate as sg
from tslearn.datasets import UCR_UEA_datasets
import os
import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

def sax_generate(size, length, data_amount):
    data = np.load('../data/'+'generate_data_'+str(data_amount)+'.npy')
    data = data.reshape(len(data), len(data[0]))
    label = np.load('../data/'+'generate_label_'+str(data_amount)+'.npy')

    # indicies = np.random.choice(data_amount, 1000, replace=False)
    # data_test = data[indicies]
    # label_test = label[indicies]
    # data = np.delete(data, indicies, axis=0)
    # label = np.delete(label, indicies, axis=0)
    test_data_amount = 1000
    data, data_test, label, label_test = train_test_split(data, label, test_size=test_data_amount, random_state=2023)
    
    symbol_size = size
    paa_length = length
    saxs = [sg.sax(elem, symbol_size, paa_length) for elem in data]
    data = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in saxs]  
    
    test_sax = [sg.sax(elem, symbol_size, paa_length) for elem in data_test]
    data_test = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in test_sax]
    # random.shuffle(data)
    return data, label, data_test, label_test


def generate_files(symbol_size, window_length, data_amount):
    path = '../data/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    data, label, test_data, test_label = sax_generate(symbol_size, window_length, data_amount)
    file = open(path+'/data.txt', 'w')
    for d in data:
        file.write(str(d)+'\n')
    file.close()
    file = open(path+'/label.txt', 'w')
    for l in label:
        file.write(str(l)+'\n')
    file.close()
    
    file = open(path+'/test_data.txt', 'w')
    for d in test_data:
        file.write(str(d)+'\n')
    file.close()
    file = open(path+'/test_label.txt', 'w')
    for l in test_label:
        file.write(str(l)+'\n')
    file.close()
    return


def amplitute_warping(data, length):
    amplitute = np.random.uniform(-0.3, 0.3)
    data = data*(1+amplitute)
    if len(data)>=length:
        data = data[:length]
    else:
        sub = abs(data[-1]-data[len(data)-2])
        comp = [data[-1]+np.random.uniform(-sub, sub) for _ in range(length-len(data))]
        data = np.vstack((data, comp))
    return data


def time_warping(data, length):
    amount = np.random.randint(-int(data.shape[0]*0.2), int(data.shape[0]*0.2))
    indices = np.random.choice(data.shape[0], abs(amount), replace=False)
    new_data = []
    if amount>0:
        for i in range(len(data)):
            if i not in indices:
                new_data.append(data[i])
            else:
                if i>0:
                    new_data.append(data[i-1])
                    new_data.append((data[i-1]+data[i])/2)
                    new_data.append(data[i])
                else:
                    new_data.append(data[i])
                    new_data.append(data[i])
    else:
        for i in range(len(data)):
            if i not in indices:
                new_data.append(data[i])
            else:
                if i<len(data)-1:
                    new_data.append((data[i+1]+data[i])/2)
                    i += 2
    data = np.array(new_data)
    if len(data)>=length:
        data = data[:length]
    else:
        sub = abs(data[-1]-data[len(data)-2])
        comp = [data[-1]+np.random.uniform(-sub, sub) for _ in range(length-len(data))]
        data = np.vstack((data, comp))
    return data

def delay_warping(data, length):
    amount = np.random.randint(1, int(data.shape[0]*0.2))
    delay = [data[0]+np.random.uniform(-abs(data[0]-data[1]), abs(data[0]-data[1])) for _ in range(amount)]
    delay = np.array(delay)
    data = np.vstack((delay, data))
    if len(data)>=length:
        data = data[:length]
    else:
        sub = abs(data[-1]-data[len(data)-2])
        comp = [data[-1]+np.random.uniform(-sub, sub) for _ in range(length-len(data))]
        comp = np.array(comp)
        data = np.vstack((data, comp))
    return data
    

def generate_more_data(data_set_name):
    data_train, label_train, data_test, label_test = UCR_UEA_datasets().load_dataset(data_set_name)
    data = np.vstack((data_train, data_test))
    label = np.hstack((label_train, label_test))
    data = data[label!=3]
    label = label[label!=3]
    for i in range(len(label)):
        if label[i] == 4:
            label[i] =3
    file = open('../data/data.npy', 'wb')
    np.save(file, data)
    file.close()
    file = open('../data/label.npy', 'wb')
    np.save(file, label) 
    file.close()   


def data_generate(data_amount):
    file = open('../data/data.npy', 'rb')
    data = np.load(file)
    file.close()
    file = open('../data/label.npy', 'rb')
    label = np.load(file)
    file.close()
    
    # data_amount = 80000
    length = 300
    generate_data = []
    generate_label = []
    index = 0
    p = [0.01, 0.33, 0.33, 0.33]
    while index<data_amount:
        i = np.random.randint(0, data.shape[0])
        d = data[i]
        l = label[i]
        opt = np.random.choice([0, 1, 2, 3], p=p)
        # print(opt)
        if opt == 0:
            gd = d
            if len(gd)>length:
                gd = gd[:length]
            else:
                # print(gd[-1], gd[len(gd)-2])
                sub = abs(gd[-1]-gd[len(gd)-2])
                comp = [gd[-1]+np.random.uniform(-sub, sub) for _ in range(length-len(gd))]
                gd = np.vstack((gd, comp))
        elif opt == 1:
            gd = amplitute_warping(d, length)
        elif opt == 2:
            gd = time_warping(d, length)
        else:
            gd = delay_warping(d, length)
        # print(gd)
        generate_data.append(gd)
        generate_label.append(l)
        index += 1
    # print(generate_data)
    # print(generate_label)
    generate_data = np.array(generate_data)
    generate_label = np.array(generate_label)
    file = open('../data/generate_data_'+str(data_amount)+'.npy', 'wb')
    np.save(file, generate_data)
    file.close()
    file = open('../data/generate_label_'+str(data_amount)+'.npy', 'wb')
    np.save(file, generate_label)


if __name__ == '__main__':
    # generate_files(4, 10, 'Trace') 
    # generate_more_data('Trace')
    data_generate(40000)
    generate_files(4, 10, 40000) 
    
    # data_amount = 40000
    # file = open('./data/Trace/generate_label_'+str(data_amount)+'.npy', 'rb')
    # data = np.load(file)
    # file.close()
    
    
    
    