import numpy as np
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')
test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')
data = np.concatenate((train_data, test_data), axis=0)
label = np.concatenate((train_label, test_label), axis=0)
# ks = KShape(n_clusters=6, verbose=True, random_state=2023)
ks = KMeans(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
y_pred = ks.fit_predict(data)

np.save('y_pred.npy', y_pred)
np.save('ks_cluster_centers.npy', ks.cluster_centers_)

# np.random.seed(2024)
# # 随机选取每个类别的100条数据
# selected_data = []
# selected_label = []
# classes = np.unique(label)  # 获取所有类别
# print(classes)
# for c in classes:
#     indices = np.where(label == c)[0]  # 获取类别为c的索引
#     plt.plot(data[indices[0]])
#     plt.plot(data[indices[10]])
#     plt.savefig('shape'+str(c)+'.png')
#     plt.cla()
#     np.random.shuffle(indices)  # 将索引打乱
#     selected_indices = indices[:50]  # 选择前100个索引
#     selected_data.append(data[selected_indices])  # 添加选定的数据
#     selected_label.append(label[selected_indices])  # 添加选定的标签

# # 将选定的数据和标签转换为numpy数组
# selected_data = np.concatenate(selected_data, axis=0)


# print(y_pred)
# print(selected_label)

plt.figure(figsize=(15, 10))
for yi in range(6):
    plt.subplot(3, 2, 1 + yi)
    for xx in data[y_pred == yi][:20]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    # plt.xticks([0, 100, 200, 300, 400])
    # plt.ylim(-4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.savefig('data_shape_correlated.png')