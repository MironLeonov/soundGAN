import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from data import VideoDataset
from torch.utils.data import DataLoader
import os

from discriminator import Discriminator
from main_generator import MainGenerator

from params import TrainParams

device = torch.device('cpu')

G = MainGenerator().to(device)
D = Discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=TrainParams.LEARNING_RATE, betas=(TrainParams.BETA1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=TrainParams.LEARNING_RATE, betas=(TrainParams.BETA1, 0.999))

# TODO: Schedulers

loss_bce = nn.BCELoss()
loss_mce = nn.MSELoss()

test_transformer = transforms.Compose([
    transforms.ToTensor(),
])

test_ds = VideoDataset(root_dir='features_data/hammer', transform=test_transformer)
loader = DataLoader(test_ds, batch_size=int(TrainParams.BATCH_SIZE))

# TODO: data splitted to train/test


d_scores = open('d_scores.txt', 'w')
g_scores = open('g_scores.txt', 'w')
os.makedirs('checkpoints', exist_ok=True)

if __name__ == '__main__':

    for epoch in range(TrainParams.EPOCHS):
        for idx, batch in enumerate(loader):
            idx += 1
            if idx == 1:
                real_mel_spec = batch[0].to(device)
                video_frames = batch[1].to(device)

                real_labels = torch.ones(TrainParams.BATCH_SIZE).to(device)
                fake_labels = torch.zeros(TrainParams.BATCH_SIZE).to(device)

                # Train Discriminator
                with torch.no_grad():
                    fake_mel_spec = G(video_frames, real_mel_spec)

                real_mel_spec_for_checking = real_mel_spec.squeeze()  # remove batch_size, not working with diff size

                fake_outputs = D(video_frames, fake_mel_spec)
                real_outputs = D(video_frames, real_mel_spec_for_checking)

                real_loss = loss_bce(real_outputs, real_labels)

                fake_loss = loss_bce(fake_outputs, fake_labels)

                D_loss = real_loss + fake_loss
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                # Train Generator

                fake_mel_spec = G(video_frames, real_mel_spec)

                with torch.no_grad():
                    fake_outputs = D(video_frames, fake_mel_spec)

                loss_from_D = loss_bce(fake_outputs, real_labels)
                loss_from_mel = loss_mce(fake_mel_spec, real_mel_spec_for_checking)

                G_loss = loss_from_D + TrainParams.ADDITIONAL_LOSS_COEFF * loss_from_mel
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                if idx % 100 == 0 or idx == len(loader):

                    d_scores.write(f'epoch: {epoch} Iteration: {idx} real_loss: {real_loss} fake_loss: {fake_loss}\n')
                    g_scores.write(
                        f'epoch: {epoch} Iteration: {idx} loss_from_D: {loss_from_D} loss_from_mel: {loss_from_mel}\n')

        if (epoch + 1) % 10 == 0:
            torch.save(G, f'checkpoints/Generator_epoch_{epoch}.pth')
            torch.save(D, f'checkpoints/Discriminator_epoch_{epoch}.pth')
            print('Model saved.')
