import numpy as np
import scipy.stats as stats
import sys
import math


symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
size_3 = [(-sys.maxsize, -0.43), (-0.43, 0.43), (0.43, sys.maxsize)]
size_4 = [(-sys.maxsize, -0.67), (-0.67, 0), (0, 0.67), (0.67, sys.maxsize)]
size_5 = [(-sys.maxsize, -0.84), (-0.84, -0.25), (-0.25, 0.25), (0.25, 0.84), (0.84, sys.maxsize)]
# size_6 = [(-sys.maxsize, -0.97), (-0.97, -0.43), (-0.43, 0), (0, 0.43), (0.43, 0.97), (0.97, sys.maxsize)]
size_6 = [(-sys.maxsize, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, sys.maxsize)]
size_7 = [(-sys.maxsize, -1.07), (-1.07, -0.57), (-0.57, -0.18), (-0.18, 0.18), (0.18, 0.57), (0.57, 1.07), (1.07, sys.maxsize)]
# size_7 = [(-sys.maxsize, -1), (-1, -0.5), (-0.5, -0.25), (-0.25, 0.25), (0.25, 0.5), (0.5, 1), (1, sys.maxsize)]
# size_8 = [(-sys.maxsize, -1.15), (-1.15, -0.67), (-0.67, -0.32), (-0.32, 0), (0, 0.32), (0.32, 0.67), (0.67, 1.15), (1.15, sys.maxsize)]
size_8 = [(-sys.maxsize, -0.99), (-0.99, -0.66), (-0.66, -0.33), (-0.33, 0), (0, 0.33), (0.33, 0.66), (0.66, 0.99), (0.99, sys.maxsize)]

size_9 = [(-sys.maxsize, -1.22), (-1.22, -0.76), (-0.76, -0.43), (-0.43, -0.14), (-0.14, 0.14), (0.14, 0.43), (0.43, 0.76), (0.76, 1.22), (1.22, sys.maxsize)]
# size_10 = [(-sys.maxsize, -1.28), (-1.28, -0.84), (-0.84, -0.52), (-0.52, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.52), (0.52, 0.84), (0.84, 1.28), (1.28, sys.maxsize)]
size_10 = [(-sys.maxsize, -1), (-1, -0.75), (-0.75, -0.5), (-0.5, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, sys.maxsize)]
size_12 = [(-sys.maxsize, -1), (-1, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
sheet = [None, None, None, size_3, size_4, size_5, size_6, size_7, size_8, size_9, size_10, None, size_12]
cutlines = {
        2 : [0],
        3 : [-0.43, 0.43],
        4 : [-0.67, 0, 0.67],
        5 : [-0.84, -0.25, 0.25, 0.84],
        # 6 : [-0.97, -0.43, 0, 0.43, 0.97],
        6 : [-1, -0.5, 0, 0.5, 1],
        7 : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        # 7 : [-1, -0.5, -0.25, 0.25, 0.5, 1],
        # 8 : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
        # 8 : [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
        8: [-0.99, -0.66, -0.33, 0, 0.33, 0.66, 0.99],
        9 : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
        # 10 : [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
        10: [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        12 : [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        }


def length_2_pattern(size):
    alphabets = symbols[:size]
    candidates = []
    for i in alphabets:
        for j in alphabets:
            if i!=j:
                candidates.append((i, j))
    return candidates

def sax(data, symbolic_size, size):
    data = stats.zscore(data)
    paa_result = []
    sequence_length = math.floor(len(data)/size)
    for i in range(sequence_length):
        paa_result.append(np.mean(data[i*size:(i+1)*size]))
    sax_result = []
    for elem in paa_result:
        # print(symbolic_value(symbolic_size, elem))
        # print(symbolic_size, elem)
        symbol_, _ = symbolic_value(symbolic_size, elem)
        sax_result.append(symbol_)
    return sax_result


def sax_delete_repeat(sax_sequence):
    sax_sequence = list(sax_sequence)
    for i in range(len(sax_sequence)-1):
        if sax_sequence[i] == sax_sequence[i+1]:
            sax_sequence[i] = ' '
    return ''.join(sax_sequence).replace(' ', '')


def symbolic_value(symbolic_size, value):
    for elem_index in range(len(sheet[symbolic_size])):
        # print(value, sheet[symbolic_size][elem_index][0], sheet[symbolic_size][elem_index][1])
        # print(value)
        if sheet[symbolic_size][elem_index][0]<=value<=sheet[symbolic_size][elem_index][1]:
            # print(symbols[elem_index], elem_index)
            return symbols[elem_index], elem_index