import numpy as np 
import matplotlib.pyplot as plt

data = np.load('../data/train_data.npy')
label = np.load('../data/train_label.npy')
label_set = np.unique(label)
center = []
for i in label_set:
    indicies = np.where(label == i)[0]
    current_data = data[indicies]
    center.append(np.mean(current_data, axis=0))
    plt.plot(np.mean(current_data, axis=0), label=i)
plt.legend()
plt.savefig('center.png')

center = np.array(center)
np.save('ground_truth_center.npy', center)
    
    