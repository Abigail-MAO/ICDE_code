import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


path = '/data/fat/maoyl/ShapeExtraction/'+'Symbols_12_04/'
index = 0
epsilon = 4
data = np.load(path+'train/patternLDP/'+str(epsilon)+'/'+str(index)+'_data.npy')
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
# ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
# centers = ks.cluster_centers_
# for i in range(len(centers)):
#     plt.plot(centers[i])
# plt.savefig('center.png')
