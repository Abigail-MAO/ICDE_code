import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

length = [200, 400, 600, 800, 1000]
    # length = [1000]
    
    # epsilons = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
para = []
for l in length:
    print("length: ", l)
    
    
    
    data = np.load('../data/train_data_'+str(l)+'.npy')
    label = np.load('../data/train_label_'+str(l)+'.npy')
    test_data = np.load('../data/test_data_'+str(l)+'.npy')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    # print(test_data.shape)
    # exit()
    test_label = np.load('../data/test_label_'+str(l)+'.npy')

    clf = RandomForestClassifier()
    # clf = KNeighborsClassifier(n_neighbors=1, metric=fast_ddtw, n_jobs=-1)
    clf.fit(data, label)
    acc = clf.score(test_data, test_label)
    print(l, acc)