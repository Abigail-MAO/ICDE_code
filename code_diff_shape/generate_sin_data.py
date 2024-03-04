import numpy as np
import matplotlib.pyplot as plt

data_amount = 10000
length = [200, 400, 600, 800, 1000]
colors = ['r', 'g', 'b', 'orange', 'purple']
fig = plt.figure(figsize=(5, 2))
for l in length:
    x = np.linspace(0, 2*np.pi, l)
    sin_y = np.sin(x)
    cos_y = np.cos(x)
    data = []
    label = []
    value_array = sin_y[:l]
    mean = np.mean(sin_y[:l])
    std = np.std(sin_y[:l])
    value_array = [(value_array[i]-mean)/std for i in range(l)]
    data.extend([value_array for _ in range(data_amount)])
    label.extend([0 for _ in range(data_amount)])
    # plt.plot(value_array, label='sin', color= colors[length.index(l)], linewidth=3)
    value_array = cos_y[:l]
    mean = np.mean(cos_y[:l])
    std = np.std(cos_y[:l])
    value_array = [(value_array[i]-mean)/std for i in range(l)]
    data.extend([value_array for _ in range(data_amount)])
    label.extend([1 for _ in range(data_amount)])
    data = np.array(data)
    label = np.array(label)
    
    plt.plot(value_array, label='cos', color= colors[length.index(l)], linewidth=3)
    plt.xticks([0, 200, 400, 600, 800, 1000])
    # plt.yticks([])
    plt.grid(True)
    # plt.legend(fontsize=20)
plt.savefig('./data/cos'+'.pdf')
    # plt.cla()
    
    # test_indices = np.random.choice(len(data), 1000, replace=False)
    # test_data = data[test_indices]
    # test_label = label[test_indices]
    # train_data = np.delete(data, test_indices, axis=0)
    # train_label = np.delete(label, test_indices, axis=0)
    # np.save('./data/train_data_'+str(l)+'.npy', train_data)
    # np.save('./data/train_label_'+str(l)+'.npy', train_label)
    # np.save('./data/test_data_'+str(l)+'.npy', test_data)
    # np.save('./data/test_label_'+str(l)+'.npy', test_label)
    
    

