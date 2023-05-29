import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.utils import save_image
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channel = base_channel

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channel * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channel * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(base_channel * 8, base_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(base_channel * 4, base_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(base_channel * 2, base_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel, base_channel, 3, 1, 0, dilation=2, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        x = self.conv_blocks(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channel):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.in_channels = in_channels
        self.base_channel = base_channel

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channel * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel * 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel * 1, base_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel * 2, base_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channel * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel * 4, base_channel * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(base_channel * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channel * 8, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        x = self.model(img).squeeze()
        return x


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    criterion = torch.nn.BCELoss()
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    plot_epoch = []
    weighted_dis = []
    weighted_gen = []
    for epoch in range(args.n_epochs):
        weighted_dis_loss = 0
        weighted_gen_loss = 0
        bar = tqdm(dataloader)
        for i, (imgs, _) in enumerate(bar):
            imgs = imgs.to(device)
            label_0 = torch.zeros(args.batch_size).to(device)
            label_1 = torch.ones(args.batch_size).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            noise = torch.randn((args.batch_size, args.latent_dim)).to(device)
            noise = noise.view(args.batch_size, args.latent_dim, 1, 1)
            fake_image = generator(noise)
            # fake_image = fake_image.to(device)
            # print("fake_image.device:", fake_image.device)
            # print("discriminator.parameters().device:", next(discriminator.parameters()).device)
            prediction_on_fake = discriminator(fake_image)
            gen_loss = criterion(prediction_on_fake, label_1)
            gen_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            prediction_on_real = discriminator(imgs)
            loss_1 = criterion(prediction_on_real, label_1)

            prediction_on_fake = discriminator(fake_image.detach())
            loss_0 = criterion(prediction_on_fake, label_0)
            loss = (loss_0 + loss_1) / 2
            loss.backward()
            optimizer_D.step()

            # print(epoch, i, gen_loss.item(), loss.item())
            # weighted loss of generator and discriminator
            weighted_gen_loss = weighted_gen_loss * (i / (i + 1.)) + gen_loss * (1. / (i + 1.))
            weighted_dis_loss = weighted_dis_loss * (i / (i + 1.)) + loss * (1. / (i + 1.))
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                save_image(fake_image[:25], './images/{}.png'.format(batches_done), nrow=5, normalize=True)
                torch.save(generator, './models/G_{}'.format(batches_done))
                torch.save(discriminator, './models/D_{}'.format(batches_done))
        plot_epoch.append(epoch)
        weighted_gen.append(weighted_gen_loss)
        weighted_dis.append(weighted_dis_loss)
        print('Epoch: {}, Generator loss: {}, Discriminator loss: {}'.format(plot_epoch[len(plot_epoch) - 1],
                                                                             weighted_gen[len(weighted_gen) - 1],
                                                                             weighted_dis[len(weighted_dis) - 1]))

    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)
    fig1.plot(plot_epoch, weighted_dis, c='red', label='discriminator loss')
    fig1.legend()
    fig2.plot(plot_epoch, weighted_gen, c='green', label='generator loss')
    fig2.legend()
    plt.savefig('/home/s11711335/gan2/plot1.jpg')
    plt.show()
    print('Done training.')


def main():
    print('GAN WORKING...')
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # grey graph, not RGB
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(100, 1, 128)
    discriminator = Discriminator(1, 128)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    print('GAN DONE')
    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


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
