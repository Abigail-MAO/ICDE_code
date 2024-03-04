import sax_generate as sg
from tslearn.datasets import UCR_UEA_datasets
import os
import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

def sax_generate(size, length, data_length):
    data = np.load('../data/'+'train_data_'+str(data_length)+'.npy')
    data = data.reshape(len(data), len(data[0]))
    label = np.load('../data/'+'train_label_'+str(data_length)+'.npy')
    
    data_test = np.load('../data/'+'test_data_'+str(data_length)+'.npy')
    data = data.reshape(len(data), len(data[0]))
    label_test = np.load('../data/'+'test_label_'+str(data_length)+'.npy')
    
    symbol_size = size
    paa_length = length
    saxs = [sg.sax(elem, symbol_size, paa_length) for elem in data]
    data = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in saxs]  
    
    test_sax = [sg.sax(elem, symbol_size, paa_length) for elem in data_test]
    data_test = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in test_sax]
    # random.shuffle(data)
    return data, label, data_test, label_test


def generate_files(symbol_size, window_length, data_length):
    path = './data/'+str(data_length)+'/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    data, label, test_data, test_label = sax_generate(symbol_size, window_length, data_length)
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
    # for symbol_size in range(4, 11):
    #         generate_files(symbol_size, 10, 40000)
    for length in [200, 400, 600, 800, 1000]:
        generate_files(4, 10, length)
    # generate_files(3, 10, 40000) 
    
    # data_amount = 40000
    # file = open('./data/Trace/generate_label_'+str(data_amount)+'.npy', 'rb')
    # data = np.load(file)
    # file.close()
    