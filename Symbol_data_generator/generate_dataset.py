import gan_trainer as gt
import pandas as pd
import gan
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

def generation(name, generator_name, amount):
    df = pd.read_csv('./data/Symbols_data.csv')
    df = df.loc[df['label'] == name]

    # real signal
    N = amount
    # real_samples =  df.sample(N).values[:, :-2].transpose()

    # synthetic signal
    g = gan.Generator()
    g.load_state_dict(torch.load('./model/'+generator_name+'generator.pth'))
    g.eval()
    d = gan.Discriminator()
    d.load_state_dict(torch.load('./model/'+generator_name+'discriminator.pth'))
    d.eval()
    trainer = gt.Trainer(
        generator=g,
        discriminator=d,
        batch_size=20,
        num_epochs=1200,
        label=name)
    fake = trainer.netG(trainer.fixed_noise)
    index = np.random.choice(fake.shape[0], N, replace=True) 
    synthetic_samples = fake.detach().cpu().squeeze(1).numpy()[index].transpose()
    synthetic_samples = synthetic_samples.reshape((synthetic_samples.shape[1], synthetic_samples.shape[0]))
    return synthetic_samples


if __name__ == '__main__':
    names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    g_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    amount = [7060, 6320, 6520, 7060, 6320, 6520]
    labels = [1, 2, 3, 4, 5, 6]
    
    for i in range(len(names)):
        d = generation(names[i], g_names[i], amount[i])
        print(d.shape)
        for j in d[:5]:
            plt.plot(j)
        plt.show()
        plt.cla()
        file = open('./data/new_data.csv', 'a+')
        for j in range(d.shape[0]):
            for k in range(d.shape[1]):
                file.write(str(d[j][k])+',')
            file.write(str(labels[i])+','+names[i]+'\n')
            # sys.exit()
        file.close()
        