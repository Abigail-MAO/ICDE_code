import matplotlib.pyplot as plt


paras = [(6, 10), (6, 15), (6, 20), (6, 25), (6, 30), (6, 35), (6, 40), (6, 45), (3, 25), (5, 25), (6, 25), (7, 25), (8, 25), (9, 25), (10, 25)]
window_length = 25
symbol_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
result = []
for symbol_size in symbol_sizes:
    sum_acc = 0
    for index in range(28):
        file = open('./result/'+str(symbol_size)+'_'+str(window_length)+'/acc_'+str(index)+'.txt', 'r')
        value = float(file.readline())
        sum_acc += value
    result.append(sum_acc/28/18)

plt.plot(symbol_sizes, result)
plt.xlabel('Symbol Size')
plt.ylabel('Adjusted Rand Index')
plt.grid(linestyle='-.')
plt.savefig('varying_symbol_size.png')




window_lengths = [10, 15, 20, 25, 30, 35, 40, 45]
symbol_size = 6
result = []
for window_length in window_lengths:
    sum_acc = 0
    for index in range(28):
        file = open('./result/'+str(symbol_size)+'_'+str(window_length)+'/acc_'+str(index)+'.txt', 'r')
        value = float(file.readline())
        sum_acc += value
    result.append(sum_acc/28/18)

plt.plot(symbol_sizes, result)
plt.xlabel('Window Length')
plt.ylabel('Adjusted Rand Index')
plt.grid(linestyle='-.')
plt.savefig('varying_window_length.png')


        