import os

symbol_size = [3, 4, 5, 6, 7, 8, 9, 10]
window_length = [5, 10, 15, 20, 25, 30, 35, 40]

result = []
for window in window_length:
    times = 36
    sum_accuracy = 0
    files = os.listdir('./result/'+str(4)+'_'+str(window))
    for path in files:
        file = open('./result/'+str(4)+'_'+str(window)+'/'+path, 'r')
        sum_accuracy += float(file.readline())
        file.close()
    print("average accuracy:", sum_accuracy/len(files)/times)
    result.append(sum_accuracy/len(files)/times)
print(result)