result = []

epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
for epsilon in epsilons:
    file = open('./result_12_04/'+str(epsilon)+"_result.txt", "r")
    lines = file.readlines()
    acc = 0
    for line in lines:
        acc += float(line)
    acc /= len(lines)
    result.append(acc)
    print(epsilon, acc)
    file.close()
print(result)