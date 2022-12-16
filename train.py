import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from data import VideoDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import librosa.display

warnings.simplefilter(action='ignore', category=UserWarning)

from discriminator import Discriminator
from main_generator import MainGenerator

from params import TrainParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = MainGenerator().to(device)
D = Discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=TrainParams.LEARNING_RATE, betas=(TrainParams.BETA1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=TrainParams.LEARNING_RATE, betas=(TrainParams.BETA1, 0.999))

loss_bce_gen = nn.BCELoss()
loss_bce_discr = nn.BCELoss()
loss_mce = nn.MSELoss()

test_transformer = transforms.Compose([
    transforms.ToTensor(),
])

test_ds = VideoDataset(root_dir='C:/Projects/soundGAN-main/features_data/hammer', transform=test_transformer)
loader = DataLoader(test_ds, batch_size=int(TrainParams.BATCH_SIZE), shuffle=True)

print('Loader initialized')

d_scores = open('d_scores.txt', 'w')
g_scores = open('g_scores.txt', 'w')
os.makedirs('checkpoints', exist_ok=True)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


os.makedirs('figures', exist_ok=True)

if __name__ == '__main__':
    print(f'Start training on {device}')

    for epoch in range(TrainParams.EPOCHS):
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        for idx, batch in progress_bar:

            idx += 1
            real_mel_spec = batch[0].to(device)
            video_frames = batch[1].to(device)

            real_labels = torch.ones(TrainParams.BATCH_SIZE).to(device)
            fake_labels = torch.zeros(TrainParams.BATCH_SIZE).to(device)


            # Train Discriminator
            set_requires_grad(G, False)
            set_requires_grad(D, True)

            with torch.no_grad():
                fake_mel_spec = G(video_frames, real_mel_spec)

            fake_outputs = D(video_frames, fake_mel_spec.detach())
            real_outputs = D(video_frames, real_mel_spec)

            real_loss = loss_bce_discr(real_outputs, real_labels)

            fake_loss = loss_bce_discr(fake_outputs, fake_labels)

            D_loss = (real_loss + fake_loss) * 0.5
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Train Generator

            set_requires_grad(G, True)
            set_requires_grad(D, False)

            fake_mel_spec = G(video_frames, real_mel_spec)

            with torch.no_grad():
                fake_outputs = D(video_frames, fake_mel_spec)

            loss_from_D = loss_bce_gen(fake_outputs, real_labels)
            loss_from_mel = loss_mce(fake_mel_spec, real_mel_spec)

            G_loss = loss_from_D + TrainParams.ADDITIONAL_LOSS_COEFF * loss_from_mel
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            if idx % 3 == 0 or idx == len(loader):
                progress_bar.set_description(f"[{epoch + 1}/{TrainParams.EPOCHS}][{idx + 1}/{len(loader)}] "
                                             f"real_loss: {real_loss} fake_loss: {fake_loss} "
                                             f"loss_from_D: {loss_from_D} loss_from_mel: {loss_from_mel} gen_loss: {G_loss}")
                d_scores.write(f'Epoch: {epoch} Iteration: {idx} real_loss: {real_loss} fake_loss: {fake_loss}\n')
                g_scores.write(
                    f'Epoch: {epoch} Iteration: {idx} loss_from_D: {loss_from_D} loss_from_mel: {loss_from_mel} gen_loss: {G_loss}\n')

                if idx % 15 == 0:
                    with torch.no_grad():
                        fake_mel_spec = G(video_frames, real_mel_spec)

                    fig, (ax1, ax2) = plt.subplots(2)

                    fake = fake_mel_spec[0].to('cpu').numpy()
                    img = librosa.display.specshow(fake, y_axis='linear', ax=ax1)
                    ax1.set(title='Fake')
                    fig.colorbar(img, ax=ax1, format="%+2.f dB")

                    real = real_mel_spec[0].to('cpu').numpy()
                    img = librosa.display.specshow(real, x_axis='time', y_axis='linear', ax=ax2)
                    ax2.set(title='Real')
                    fig.colorbar(img, ax=ax2, format="%+2.f dB")

                    plt.savefig(f'figures/{epoch}-{idx}.png')
                    plt.close()

        if (epoch + 1) % 5 == 0:
            torch.save(G, f'checkpoints/Generator_epoch_{epoch}.pth')
            torch.save(D, f'checkpoints/Discriminator_epoch_{epoch}.pth')
            print(f'{epoch} Model saved.')
