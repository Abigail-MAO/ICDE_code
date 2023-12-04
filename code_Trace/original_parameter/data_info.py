from tslearn.datasets  import  UCR_UEA_datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

def data_generate(size, length):
    print("here")
    tsd = UCR_UEA_datasets()
    dataset_name = 'Trace'
    print(dataset_name) 
    data_train, label_train, data_test, label_test = tsd.load_dataset(dataset_name)
    # data_train = data_train[label_train[:]]
    # label_train = label_train[label_train[:]]
    # data_test = data_test[label_test[:]]
    # label_test = label_test[label_test[:]]
    # data = np.vstack((data_train, data_test))
    # label = np.hstack((label_train, label_test))
    label = label_train
    # print(len(data))
    files = os.listdir('./figure/')
    for path in files:
        os.remove('./figure/'+path)
    # label_ = set()
    
    print(label)
    class_amount = 2
    for c in [1, 2, 3, 4]:
        count = 0
        index = 0
        while count<5:
            print(count)
            index+= 1
            if label[index]==c:
                count += 1
                plt.plot(stats.zscore(data_train[index]))
                plt.savefig('./figure/'+str(label[index])+'_'+str(count)+'.png')
                plt.cla()
        
        
    
    
    # symbol_size = size
    # paa_length = length
    # saxs = [gd.sax(elem, symbol_size, paa_length) for elem in data]
    # sequences = [gd.sax_delete_repeat(sax_sequence) for sax_sequence in saxs]  
    # data = []
    # labels = []
    # for _ in range(1):
    #     data.extend(sequences)
    #     labels.extend(label)
    # print(labels[0])
    # count = 0
    # index = 0
    # while count<5:
    #     if labels[index]==1:
    #         print(data[index])
    #         count += 1
    #     index += 1
    # print()
    # count = 0
    # index = 0
    # while count<5:
    #     if labels[index]==2:
    #         print(data[index])
    #         count += 1
    #     index += 1


if __name__=='__main__':
    # data_generate(4, 5)
    from tslearn.datasets import CachedDatasets
    data, label, X_test, y_test = CachedDatasets().load_dataset("Trace")
    # data = np.load('../data/data.npy')
    # label = np.load('../data/label.npy')
    class_label = [False, False, False, False]
    class_data = [None, None, None, None]
    for i in range(50, len(data)):
        if not class_label[label[i]-1]:
            class_data[label[i]-1] = data[i]
            class_label[label[i]-1] = True
        if False not in class_label:
            break
    class_data = np.array(class_data[:3])
    file = open('./class_data.npy', 'wb')
    np.save(file, class_data)
    
    import matplotlib.pyplot as plt
    
    plt.plot(class_data[2])
    plt.savefig('class3.png')