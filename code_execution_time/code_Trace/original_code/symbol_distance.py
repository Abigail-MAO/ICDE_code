import sys
import numpy as np


symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
size_3 = [(-sys.maxsize, -0.43), (-0.43, 0.43), (0.43, sys.maxsize)]
size_4 = [(-sys.maxsize, -0.67), (-0.67, 0), (0, 0.67), (0.67, sys.maxsize)]
size_5 = [(-sys.maxsize, -0.84), (-0.84, -0.25), (-0.25, 0.25), (0.25, 0.84), (0.84, sys.maxsize)]
size_6 = [(-sys.maxsize, -0.97), (-0.97, -0.43), (-0.43, 0), (0, 0.43), (0.43, 0.97), (0.97, sys.maxsize)]
size_7 = [(-sys.maxsize, -1.07), (-1.07, -0.57), (-0.57, -0.18), (-0.18, 0.18), (0.18, 0.57), (0.57, 1.07), (1.07, sys.maxsize)]
size_8 = [(-sys.maxsize, -1.15), (-1.15, -0.67), (-0.67, -0.32), (-0.32, 0), (0, 0.32), (0.32, 0.67), (0.67, 1.15), (1.15, sys.maxsize)]
size_9 = [(-sys.maxsize, -1.22), (-1.22, -0.76), (-0.76, -0.43), (-0.43, -0.14), (-0.14, 0.14), (0.14, 0.43), (0.43, 0.76), (0.76, 1.22), (1.22, sys.maxsize)]
size_10 = [(-sys.maxsize, -1.28), (-1.28, -0.84), (-0.84, -0.52), (-0.52, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.52), (0.52, 0.84), (0.84, 1.28), (1.28, sys.maxsize)]
sheet = [None, None, None, size_3, size_4, size_5, size_6, size_7, size_8, size_9, size_10]
cutlines = {
        2 : [0],
        3 : [-0.43, 0.43],
        4 : [-0.67, 0, 0.67],
        5 : [-0.84, -0.25, 0.25, 0.84],
        6 : [-0.97, -0.43, 0, 0.43, 0.97],
        7 : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        8 : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
        9 : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
        10 : [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]
        }

symbol_distance = {}

def initalize_symbol_distance(size):
    alphabet = symbols[:size]
    for i in range(size):
        s = alphabet[i]
        for j in range(size):
            t = s+alphabet[j]
            global symbol_distance
            symbol_distance[t] = 0
    return 

def two_symbols_distance(s1, s2):
    global symbol_distance
    distance = 0
    if s1 != s2:
        min_index = min(symbols.index(s1), symbols.index(s2))
        max_index = max(symbols.index(s1), symbols.index(s2))
        for i in range(min_index, max_index):
            distance += symbol_distance[symbols[i]+symbols[i+1]]
    symbol_distance[s1+s2] = distance
    return 

def consecutive_symbols_distance(size):
    global symbol_distance
    distance_array = cutlines[size]
    if size%2 == 0:
        for i in range(size//2):
            symbol_distance[symbols[i]+symbols[i+1]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[i+1]+symbols[i]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[size-1-i-1]+symbols[size-1-i]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[size-1-i]+symbols[size-1-i-1]] = distance_array[i+1]-distance_array[i]
        symbol_distance[symbols[size//2-1]+symbols[size//2]] = symbol_distance[symbols[size//2-2]+symbols[size//2-1]]
    else:
        for i in range((size-1)//2):
            symbol_distance[symbols[i]+symbols[i+1]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[i+1]+symbols[i]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[size-1-i-1]+symbols[size-1-i]] = distance_array[i+1]-distance_array[i]
            symbol_distance[symbols[size-1-i]+symbols[size-1-i-1]] = distance_array[i+1]-distance_array[i]
        symbol_distance[symbols[(size-1)//2-1]+symbols[(size-1)//2]] = symbol_distance[symbols[(size-1)//2-1]+symbols[(size-1)//2]]
        symbol_distance[symbols[(size-1)//2]+symbols[(size-1)//2+1]] = symbol_distance[symbols[(size-1)//2-1]+symbols[(size-1)//2]]
    # print(symbol_distance)
    return

def get_distance(size):
    global symbol_distance
    initalize_symbol_distance(size)
    consecutive_symbols_distance(size)
    for i in range(size):
        for j in range(size):
            if symbol_distance[symbols[i]+symbols[j]] == 0:
                two_symbols_distance(symbols[i], symbols[j])
    for i in range(size):
        symbol_distance[symbols[i]+symbols[i]] = min(list(filter(lambda x: x>0, symbol_distance.values())))
        symbol_distance[''+symbols[i]] = min(list(filter(lambda x: x>0, symbol_distance.values())))
        symbol_distance[symbols[i]+''] = min(list(filter(lambda x: x>0, symbol_distance.values())))
    return symbol_distance

def similar_match(symbol_distance, str1, str2, weight_insert=1, weight_delete=1, weight_sub=1):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * weight_delete
        ops[i][0] = 'D'
    for j in range(n + 1):
        dp[0][j] = j * weight_insert
        ops[0][j] = 'I'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j==1:
                left_char = ''
            else:
                left_char = str2[j-2]
            if j==n:
                right_char = ''
            else:
                right_char = str2[j]
            weight = min(symbol_distance[left_char+str1[i-1]], symbol_distance[str1[i-1]+right_char])
            delete_cost = dp[i - 1][j] + weight
            insert_cost = dp[i][j - 1] + weight
            sub_cost = dp[i - 1][j - 1] + (0 if str1[i - 1] == str2[j - 1] else weight)
            dp[i][j] = min(delete_cost, insert_cost, sub_cost)
            if dp[i][j] == delete_cost:
                ops[i][j] = 'D'
            elif dp[i][j] == insert_cost:
                ops[i][j] = 'I'
            else:
                if str1[i - 1] == str2[j - 1]:
                    ops[i][j] = 'N'
                else:
                    ops[i][j] = 'S'
    # for op in ops:
    #     print(op)
    # i = m
    # j = n
    # operations = []
    # while i > 0 or j > 0:
    #     if ops[i][j] == "N":
    #         i -= 1
    #         j -= 1
    #     elif ops[i][j] == "I":
    #         j -= 1
    #         operations.append(("I", j, str2[j]))
    #     elif ops[i][j] == "D":
    #         i -= 1
    #         operations.append(("D", i, str1[i]))
    #     else:  # ops[i][j] == "replace"
    #         i -= 1
    #         j -= 1
    #         operations.append(("R", i, str1[i], j, str2[j]))
    # print(operations[::-1])
    i = m
    j = n
    while ops[i][j] == 'D':
        i -= 1
    return dp[i][j]

def similar_match_dtw(symbol_dist, str1, str2):
    m, n = len(str1), len(str2)
    dtw = np.zeros((m+1, n+1))
    
    # Initializing the first row and column of the DTW matrix
    for i in range(1, m+1):
        dtw[i][0] = np.inf
    for j in range(1, n+1):
        dtw[0][j] = np.inf
    dtw[0][0] = 0
    
    # Filling in the DTW matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = symbol_dist[str1[i-1]+str2[j-1]]
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[m][n]



if __name__ == '__main__':
    # a = similar_match(get_distance(4), 'cdac', 'cdab')
    # b = similar_match(get_distance(4), 'cdac', 'cdad')
    a = similar_match_dtw(get_distance(3), 'b', 'a')
    b = similar_match_dtw(get_distance(3), 'b', 'c')
    print(a, b)