import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from audio_loader import AudioLoader
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def spect_loader3(path):
    y, sr = sf.read(path)

    # matching all the arrays to be 2 seconds
    if len(y) < sr:
        y = np.pad(y, (0, sr * 2 - len(y)), 'constant')
    elif len(y) > sr:
        y = y[:sr]

    y = y[::2]

    y = torch.FloatTensor(y)

    mean = y.mean()
    std = y.std()
    if std != 0:
        y.add_(-mean)
        y.div_(std)

    return y


num_epochs = 10
batch_size = 128
learning_rate = 1e-3

audio_loader_train = AudioLoader('data/train')
audio_loader_dev = AudioLoader('data/dev', test_mode=True)
data_loader_tr = DataLoader(audio_loader_train, batch_size=batch_size, shuffle=False)
data_loader_vl = DataLoader(audio_loader_dev, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(24000, 20000))
        self.decoder = nn.Sequential(
            nn.Linear(20000, 24000),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

print('---------- Networks architecture -------------')
utils.print_network(model)
print('-----------------------------------------------')

for epoch in range(num_epochs):
    global loss
    model.train()
    for iter, (x_noisy, x) in enumerate(data_loader_tr):
        sound = x_noisy
        sound = sound.view(sound.size(0), -1)
        sound = Variable(sound)
        # ===================forward=====================
        output = model(sound)
        loss = criterion(output, x)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.tolist()))
    # model.eval()
    # x = spect_loader3("data/dev/noisy/p232_001.wav")
    # x_clean = spect_loader3("data/dev/clean/p232_001.wav")
    # recon_x = model.forward(x)
    # out_series = pd.Series([x[0] for x in recon_x.tolist()[0][0][0]])
    # plt.interactive(True)
    # out_series.plot()
    # plt.show(block=True)
    # x_noisy_series = pd.Series(x)
    # plt.interactive(True)
    # x_noisy_series.plot()
    # plt.show(block=True)
    # x_clean_series = pd.Series(x_clean)
    # plt.interactive(True)
    # x_clean_series.plot()
    # plt.show(block=True)
    #
    # x = spect_loader3("data/train/noisy/p226_001.wav")
    # x_clean = spect_loader3("data/train/clean/p226_001.wav")
    # recon_x = model.forward(x)
    # out_series = pd.Series([x[0] for x in recon_x.tolist()[0][0][0]])
    # plt.interactive(True)
    # out_series.plot()
    # plt.show(block=True)
    # x_noisy_series = pd.Series(x)
    # plt.interactive(True)
    # x_noisy_series.plot()
    # plt.show(block=True)
    # x_clean_series = pd.Series(x_clean)
    # plt.interactive(True)
    # x_clean_series.plot()
    # plt.show(block=True)
torch.save(model.state_dict(), './sim_autoencoder.pth')
