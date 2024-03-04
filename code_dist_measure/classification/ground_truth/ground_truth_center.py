import numpy as np 
from tslearn.clustering import KShape
import matplotlib.pyplot as plt

# data = np.load('../data/train_data.npy')
# indicies = np.random.choice(data.shape[0], 300, replace=False)
# selected_data = data[indicies]
# labels = np.load('../data/train_label.npy')

# label_ = 1
# for i in range(len(data)):
#     if labels[i] == label_:
#         plt.plot(data[i], label=label_)
#         label_ += 1
#     if label_ == 4:
#         break
# plt.legend()
# plt.savefig('ground_truth.png')
# plt.cla()

# ks = KShape(n_clusters=3, random_state=2024).fit(selected_data)
# centers = ks.cluster_centers_
# np.save('center.npy', centers)
# for i in range(len(centers)):
#     plt.plot(centers[i])
# plt.savefig('center.png')

centers = np.load('center.npy')
for i in range(len(centers)):
    plt.plot(centers[i], label=i+1)
plt.legend()
plt.savefig('center_plot.png')
plt.cla()

ground_truth_center = [centers[1], centers[0], centers[2]]
for i in range(len(ground_truth_center)):
    plt.plot(ground_truth_center[i], label=i+1)
plt.legend()
plt.savefig('center_plot1.png')
np.save('ground_truth_center.npy', ground_truth_center)


