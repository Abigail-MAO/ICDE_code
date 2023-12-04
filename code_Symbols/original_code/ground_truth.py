from  tslearn.piecewise import  OneD_SymbolicAggregateApproximation
from tslearn.datasets import CachedDatasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.kmedians import kmedians
import numpy as np
from sklearn.cluster import KMeans
import sys
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import rand_score as rd
from sklearn.metrics.cluster import adjusted_rand_score as ar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tslearn.clustering import TimeSeriesKMeans
import sys
from tslearn.clustering import KShape


# data = np.load('../data/data.npy')
# label = np.load('../data/label.npy')
# X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
# print(X_train.shape, y_train.shape)

X_train = np.load('../data/data.npy')
y_train = np.load('../data/label.npy')
print(set(y_train))
# print(y_train)
# sys.exit()
km = TimeSeriesKMeans(n_clusters=5, verbose=True)
y_pred = km.fit_predict(X_train)
print(nmi(y_train, y_pred))
print(rd(y_train, y_pred))
print(ar(y_train, y_pred))
file = open('center.txt', 'w')
for i in range(5):
    for d in range(len(X_train[0])-1):
        file.write(str(km.cluster_centers_[i][d][0])+',')
    file.write(str(km.cluster_centers_[i][-1])+'\n')
sys.exit()


# file = open('center.txt', 'w')
# file.write('[')
# for i in range(len(X_train[26])-1):
#         file.write(str(X_train[26][i])+',')
# file.write(str(X_train[26][-1])+']\n')
# for i in range(len(X_train[1])-1):
#         file.write(str(X_train[1][i])+',')
# file.write(str(X_train[1][-1])+']\n')
# sys.exit()

# # print(y_train)
# # print(X_train.shape, y_train.shape)
for symbol_size in range(4, 11):
    one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=10,
            alphabet_size_avg=symbol_size)
    one_d_sax_data = one_d_sax.fit_transform(X_train)
    # one_d_sax_data = one_d_sax_data.reshape((one_d_sax_data.shape[0], -1))
    distance = [[0 for i in range(len(X_train))] for j in range(len(X_train))]
    for i in range(len(X_train)):
        for j in range(i+1, len(X_train)):
            # print(i,j)
            dist = one_d_sax.distance(X_train[i], X_train[j])
            # print(dist)
            # if i>10:
            #     sys.exit()
            distance[i][j]  = dist
            distance[j][i] = distance[i][j]
    # print(distance)
    distance = np.array(distance)
    file = open('distance_500.npy', 'wb')
    np.save(file, distance)

    # distance = np.load('distance.npy')

    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(one_d_sax_data, y_train)

    # test_data = one_d_sax.transform(X_test)
    # test_data = test_data.reshape((test_data.shape[0], -1))
    # y_pred = knn.predict(test_data)
    # print(acc(y_test, y_pred))

    k = 5
    # choose medoid 2 and 4 in your C1 and C2 because min(D) in their cluster
    initial_medoids = [0,1,2,3,4]
    kmedoids_instance = kmedoids(distance, initial_medoids, data_type = 'distance_matrix')
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    centers = kmedoids_instance.get_medoids()
    # print(y_train[centers[0]], y_train[centers[1]])

    label = []
    for i in range(len(X_train)):
        dist_array = []
        for j in range(5):
            dist_array.append(one_d_sax.distance(X_train[i], X_train[centers[j]]))
        index = np.argmin(dist_array)
        label.append(y_train[centers[index]])
    label = np.array(label)
    print(symbol_size, rd(y_train, label))
    print(symbol_size, "accuracy:", acc(y_train, label))
    print(label)
        
# # one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=3,
# #         alphabet_size_avg=2, alphabet_size_slope=2, sigma_l=1.)
# # data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
# # # one_d_sax_data = one_d_sax.fit_transform(data)
# # print(one_d_sax.distance(data[0], data[1]))

