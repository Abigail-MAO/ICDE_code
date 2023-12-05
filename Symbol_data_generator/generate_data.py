import gan_trainer as gt
import pandas as pd
import gan
import torch
import numpy as np
import matplotlib.pyplot as plt


name = 'Fusion of paced and normal'
df = pd.read_csv('./data/ECGdata.csv')
df = df.loc[df['label'] == name]

# real signal
N = 3
real_samples =  df.sample(N).values[:, :-2].transpose()

# synthetic signal
g = gan.Generator()
g.load_state_dict(torch.load('generator.pth'))
d = gan.Discriminator()
# d.load_state_dict(torch.load('discriminator.pth'))
trainer = trainer = gt.Trainer(
    generator=g,
    discriminator=d,
    batch_size=96,
    num_epochs=3000,
    label=name)
fake = trainer.netG(trainer.fixed_noise)
index = np.random.choice(fake.shape[0], N, replace=False) 
synthetic_samples = fake.detach().cpu().squeeze(1).numpy()[index].transpose()

fig, axs = plt.subplots(1, 2, figsize=(15, 4))


axs[0].plot(real_samples, c='#007FFF')
axs[0].set_title("Real", fontsize= 12, weight="bold")


axs[1].plot(synthetic_samples, c="crimson")
axs[1].set_title("Synthetic", fontsize= 12, weight="bold")

plt.suptitle('class '+name, fontsize=18, y=1.05, weight="bold")
plt.tight_layout()
plt.savefig(name+'.png', facecolor='w', edgecolor='w', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1)
# plt.savefig('Artial Premature.svg', facecolor='w', edgecolor='w', format='svg',
#         transparent=False, bbox_inches='tight', pad_inches=0.1)