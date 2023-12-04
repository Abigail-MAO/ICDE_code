import os

symbol_size = [3, 4, 5, 6, 7, 8, 9, 10]
window_length = [10, 15, 20, 25, 30, 35, 40, 45]

result = []
for symbol in symbol_size:
    times = 18
    sum_accuracy = 0
    files = os.listdir('./result/'+str(symbol)+'_'+str(25))
    for path in files:
        file = open('./result/'+str(symbol)+'_'+str(25)+'/'+path, 'r')
        sum_accuracy += float(file.readline())
        file.close()
    print("average accuracy:", sum_accuracy/len(files)/times)
    result.append(sum_accuracy/len(files)/times)
print(result)