import numpy as np
import math


def read_files(file_name):
    file = open(file_name, 'r', encoding='UTF-8-sig')
    line = file.readline()
    line = file.readline()
    time_0 = float(line.split(',')[0])
    value = []
    time = []
    while line:
        value.append(int(line.split(',')[1]))
        time.append((float(line.split(',')[0])-time_0)/1000)
        line = file.readline()
    mean_value = np.mean(value)
    std_value = np.std(value)
    data = [(elem-mean_value)/std_value for elem in value]
    # print(np.mean(data), np.std(data))
    # print(max(data), min(data))
    return time, data
