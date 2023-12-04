import matplotlib.pyplot as plt
import numpy as np
import original_code.sax_generate as sax_generate


value = {'a':-1.2, 'b':-0.7, 'c':-0.215, 'd':0.215, 'e':0.7, 'f':1.2}
original_result = ['abcdef', 'abcef', 'babced', 'dcadfc', 'fdbdfe', 'fedbab']


# for s in original_result:
#     for i in range(len(s)):
#         plt.plot([j for j in range(i*8, i*8+8)], [value[s[i]] for _ in range(8)], color='blue', linewidth=3)
#     plt.savefig('cluster_result/'+str(original_result.index(s))+s+'.png')
#     plt.cla()


# data = np.load('./data/train_data.npy')
# label = np.load('./data/train_label.npy')
# draw = [True for i in range(7)]
# draw[0] = False
# for i in range(len(label)):
#     if draw[label[i]]:
#         file = open('./cluster_result/'+str(label[i])+'.npy', 'wb')
#         np.save(file, data[i])
#         file.close()
        
#         print(sax_generate.sax(data[i], 6, 25), label[i])
#         plt.plot(data[i], color='blue', linewidth=3)
#         plt.savefig('cluster_result/'+str(label[i])+'.png')
#         plt.cla()
#         draw[label[i]] = False
#     if not any(draw):
#         break


# patternLDP_result = np.load('./cluster_result/patternLDP.npy')
# for i in patternLDP_result:
#     plt.plot(i, linewidth=1)
#     plt.savefig('cluster_result/patternLDP.png')
# plt.cla()

from tslearn.clustering import KShape
from sklearn.metrics.cluster import adjusted_rand_score as ari
train_data = np.load('./data/train_data.npy')
test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')
ks = KShape(n_clusters=6, tol=0.01, max_iter=100, verbose=False, random_state=2023).fit(train_data)
y_pred = ks.predict(test_data)
print(ari(y_pred, test_label))