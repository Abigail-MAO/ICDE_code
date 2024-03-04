import os

# dist = [0 for i in range(6)]
# epsilon = 4
# files = os.listdir('./result/'+str(epsilon))
# for path in files:
#     file = open('./result/'+str(epsilon)+'/'+path, 'r')
#     line = file.readline()
#     line = line[1:len(line)-2]
#     line = list(map(float, line.split(", ")))
#     for i in range(6):
#         dist[i] += line[i]
#     file.close()
#     #     print("average accuracy:", sum_accuracy/len(files)/times)
#     #     result.append(sum_accuracy/len(files)/times)
# for i in range(6):
#     dist[i] = dist[i]/len(files)/18
# print(dist)
#     # file = open('result.npy', 'wb')
#     # np.save(file, result)
#     # file.close()

dist = [0 for i in range(6)]
epsilon = 4
files = os.listdir('./result_tp/'+str(epsilon))
for path in files:
    file = open('./result_tp/'+str(epsilon)+'/'+path, 'r')
    line = file.readline()
    line = line[1:len(line)-2]
    line = list(map(float, line.split(", ")))
    for i in range(6):
        dist[i] += line[i]
    file.close()
    #     print("average accuracy:", sum_accuracy/len(files)/times)
    #     result.append(sum_accuracy/len(files)/times)
for i in range(6):
    dist[i] = dist[i]/len(files)
print(dist)
    # file = open('result.npy', 'wb')
    # np.save(file, result)
    # file.close()