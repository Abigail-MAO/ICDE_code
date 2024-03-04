from fastdtw import fastdtw

a = [0.1, 0.2, 0.3, 0.4, 0.5]
b = [0.2, 0.3, 0.4, 0.5, 0.6]
distance, path = fastdtw(a, b)
print(distance)