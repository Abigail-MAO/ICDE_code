import numpy as np


epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
result = []
for epsilon in epsilons:
    acc = 0
    file = open('./result_12_04/'+str(epsilon)+"_result.txt", "r")
    lines = file.readlines()
    file.close()
    for line in lines:
        acc += float(line.strip('\n'))
    result.append(acc/len(lines))
file = open('result_12_04.npy', 'wb')
np.save(file, result)