import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from tqdm import tqdm

from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, discriminator, generator, optimizer_g, optimizer_d):
    criterion = torch.nn.BCELoss()

    for epoch in range(args.n_epochs):
        bar = tqdm(dataloader)
        g_loss, d_loss = None, None
        for i, (real_images, _) in enumerate(bar):
            real_images = real_images.to(device)
            size = real_images.shape[0]
            # 梯度置零
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            # create labels
            real_labels = torch.ones(size, dtype=torch.float, device=device)
            artificial_labels = torch.zeros(size, dtype=torch.float, device=device)
            # Train Generator
            z = torch.FloatTensor(np.random.normal(0, 1, (size, args.latent_dim))).to(device)
            artificial_images = generator(z)
            artificial_scores = discriminator(artificial_images)

            g_loss = criterion(artificial_scores, real_labels)
            g_loss.backward()
            optimizer_g.step()
            # Train Discriminator
            artificial_scores = discriminator(artificial_images.detach())
            real_scores = discriminator(real_images)

            d_loss_artificial = criterion(artificial_scores, artificial_labels)
            d_loss_real = criterion(real_scores, real_labels)
            d_loss = (d_loss_artificial + d_loss_real) / 2
            d_loss.backward()
            optimizer_d.step()

            # Save Images
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28), filename, number of rows, normalize)
                # to save the generated images, e.g.:
                save_image(artificial_images.unsqueeze(1)[:25], 'gan_images/{}.png'.format(batches_done), nrow=5,
                           normalize=True)
        # print info
        print("Epoch: {}/{}, G_Loss: {:.4f}, D_Loss {:.4f}".format(epoch,
                                                                   args.n_epochs,
                                                                   g_loss,
                                                                   d_loss,
                                                                   ))


def main():
    # Create output image directory
    os.makedirs('gan_images', exist_ok=True)
    os.makedirs('interpolations_gan', exist_ok=True)

    # load data, Normalize中参数设为(0.5,)，因为MNIST数据集为灰度图而非RGB，故为一维数据
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = SimpleGenerator(args.latent_dim)
    discriminator = SimpleDiscriminator()
    generator, discriminator = generator.to(device), discriminator.to(device)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_generator, optimizer_discriminator)

    # You can save your generator here to re-use it to generate images for your report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
