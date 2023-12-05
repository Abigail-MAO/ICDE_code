import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
import gan
import gan_trainer as gt





    
if __name__ == '__main__':
#     df_mitbih = pd.read_csv('./data/Symbols.csv', header=None)
#     df_mitbih.rename(columns={398: 'class'}, inplace=True)
#     id_to_label = {
#     1: "class 1", # 181
#     2: "class 2", # 162
#     3: "class 3", # 167
#     4: "class 4", # 181
#     5: "class 5", # 162
#     6: "class 6"  # 167
# }
#     df_mitbih['label'] = df_mitbih.iloc[:, -1].map(id_to_label)
#     print(df_mitbih['label'].value_counts())
    
    
#     df_mitbih.to_csv('./data/Symbols_data.csv', index=False)
#     print(df_mitbih['label'].value_counts())
    
    g = gan.Generator()
    d = gan.Discriminator() 
    trainer = gt.Trainer(
    generator=g,
    discriminator=d,
    batch_size=20,
    num_epochs=4500,
    label='class 6',
    amount=6520) # class 4 finished
    trainer.run()

# class 1: 7060 2400
# class 2: 6320 2100
# class 3: 6520 2200
# class 4: 7060 1500
# class 5: 6320 1500
# class 6: 6520 1500