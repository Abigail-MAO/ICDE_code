import sys


def dtw(array_a, array_b):
    distance = []
    swap = []
    swap.append(0)
    for _ in range(len(array_b)):
        swap.append(sys.maxsize )
    distance.append(swap)
    for i in range(len(array_a)):
        swap = []
        swap.append(sys.maxsize)
        for j in range(len(array_b)):
            swap.append((array_b[j]-array_a[i])*(array_b[j]-array_a[i]))
        distance.append(swap)
    min_i = 0
    min_j = 0
    for i in range(1, len(array_a)+1):
        for j in range(1, len(array_b)+1):
            if distance[i-1][j-1] > distance[i][j-1]:
                min_i = i
                min_j = j-1
            if distance[min_i][min_j] > distance[i-1][j]:
                min_i = i-1
                min_j = j

            distance[i][j] += distance[i][j]+distance[min_i][min_j]

    return distance[len(array_a)][len(array_b)]



