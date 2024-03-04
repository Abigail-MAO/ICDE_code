# file = open('./patternLDP_result/4_result.txt')
# lines = file.readlines()
# dist = [0 for i in range(6)]
# for line in lines:
#     s = line[1:len(line)-2]
#     s = list(map(float, s.split(',')))
#     for i in range(6):
#         dist[i] += s[i]
# for i in range(6):
#     dist[i] = dist[i]/500
# print(dist)


file = open('./patternLDP_result/4_result_tp.txt')
lines = file.readlines()
dist = [0 for i in range(6)]
for line in lines:
    s = line[1:len(line)-2]
    s = list(map(float, s.split(',')))
    for i in range(6):
        dist[i] += s[i]
for i in range(6):
    dist[i] = dist[i]/28
print(dist)
    