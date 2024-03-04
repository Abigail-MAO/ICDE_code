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
    # saxs = [sg.sax(elem, symbol_size, paa_length) for elem in data]
    # data = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in saxs]  
    
    
    saxs = [sg.sax(elem, symbol_size, paa_length) for elem in data]
    print(saxs[0])
    data = []
    for sax_sequence in saxs:
        data.append([sax_sequence[i] for i in range(0, len(sax_sequence), 1)])
    data = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in data]  
    
    # test_sax = [sg.sax(elem, symbol_size, paa_length) for elem in data_test]
    # data_test = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in test_sax]
    
    test_sax = [sg.sax(elem, symbol_size, paa_length) for elem in data_test]
    data_test = []
    for sax_sequence in test_sax:
        data_test.append([sax_sequence[i] for i in range(0, len(sax_sequence), 1)])
    data_test = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in data_test]
    # random.shuffle(data)
    return data, label, data_test, label_test


def generate_files(symbol_size, window_length, data_amount):
    path = './data/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    data, label, test_data, test_label = sax_generate(symbol_size, window_length, data_amount)
    file = open(path+'/'+str(symbol_size)+'_'+str(window_length)+'_'+'data.txt', 'w')
    for d in data:
        file.write(str(d)+'\n')
    file.close()
    file = open(path+'/'+str(symbol_size)+'_'+str(window_length)+'_'+'label.txt', 'w')
    for l in label:
        file.write(str(l)+'\n')
    file.close()
    
    file = open(path+'/'+str(symbol_size)+'_'+str(window_length)+'_'+'test_data.txt', 'w')
    for d in test_data:
        file.write(str(d)+'\n')
    file.close()
    file = open(path+'/'+str(symbol_size)+'_'+str(window_length)+'_'+'test_label.txt', 'w')
    for l in test_label:
        file.write(str(l)+'\n')
    file.close()
    return


if __name__ == '__main__':
    # generate_files(4, 10, 'Trace') 
    # generate_more_data('Trace')
    # data_generate(40000)
    generate_files(8, 1, 40000) 
    
    # data_amount = 40000
    # file = open('./data/Trace/generate_label_'+str(data_amount)+'.npy', 'rb')
    # data = np.load(file)
    # file.close()
    
    
    
    