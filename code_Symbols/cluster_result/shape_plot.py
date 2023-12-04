import matplotlib.pyplot as plt
import numpy as np

value = {'a':-1.2, 'b':-0.7, 'c':-0.215, 'd':0.215, 'e':0.7, 'f':1.2}
original_result = ['abcdef', 'abceff', 'babced', 'dcadfc', 'fdbdfe', 'fedbab']

fig, axs = plt.subplots(6, 3, sharex=True)


for j  in [0, 1, 2]:
    for i in range(len(original_result[j])-1):
        axs[j, 2].plot([k for k in range(i*66, i*66+66)], [value[original_result[j][i]] for _ in range(66)], color='#98ea8a')
        axs[j, 2].plot([i*66+66, i*66+66], [value[original_result[j][i]], value[original_result[j][i+1]]], color='#98ea8a')
    axs[j, 2].plot([k for k in range((len(original_result[j])-1)*66, (len(original_result[j])-1)*66+66)], [value[original_result[j][-1]] for _ in range(66)], color='#98ea8a')
    axs[j, 2].set_ylim([-1.5, 1.5])

length = 42
for j  in [5]:
    for i in range(len(original_result[j])-1):
        axs[j, 2].plot([k for k in range(i*length, i*length+length)], [value[original_result[j][i]] for _ in range(length)], color='#98ea8a')
        axs[j, 2].plot([i*length+length, i*length+length], [value[original_result[j][i]], value[original_result[j][i+1]]], color='#98ea8a')
    axs[j, 2].plot([k for k in range((len(original_result[j])-1)*length, (len(original_result[j])-1)*length+length)], [value[original_result[j][-1]] for _ in range(length)], color='#98ea8a')
    # axs[j, 2].set_ylim([-0.5, 0.5])
    
    
    
length = 25
for j  in [3, 4]:
    for i in range(len(original_result[j])-1):
        axs[j, 2].plot([k for k in range(i*length, i*length+length)], [value[original_result[j][i]] for _ in range(length)], color='#98ea8a')
        axs[j, 2].plot([i*length+length, i*length+length], [value[original_result[j][i]], value[original_result[j][i+1]]], color='#98ea8a')
    axs[j, 2].plot([k for k in range((len(original_result[j])-1)*length, (len(original_result[j])-1)*length+length)], [value[original_result[j][-1]] for _ in range(length)], color='#98ea8a')
    axs[j, 2].set_ylim([-2, 2])

       

for i in range(6):
    class_data = np.load(str(i+1)+'.npy')
    axs[i, 0].plot(class_data, color='#fb90a4')
    

pattern_data = np.load('patternLDP.npy')
for i in range(6):
    axs[i, 1].plot(pattern_data[i], color='#8795e2')
    

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])




# improved result ['cdabc', 'dcabc', 'abcdc']







plt.text(-850, -5, 'Ground Truth', ha='center', va='center', fontsize=13)
plt.text(-350, -5, 'PatternLDP', ha='center', va='center', fontsize=13)
plt.text(200, -5, 'PrivShape', ha='center', va='center', fontsize=13)

plt.ylim(-3, 4)
plt.yticks([])
plt.xticks([])
plt.savefig('shape.pdf')
