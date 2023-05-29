import torch.nn as nn


class SimpleGenerator(nn.Module):
    """
    Construct generator. You should experiment with your model, but the following is a good start:
        Linear args.latent_dim -> 128
        LeakyReLU(0.2)
        Linear 128 -> 256
        Bnorm
        LeakyReLU(0.2)
        Linear 256 -> 512
        Bnorm
        LeakyReLU(0.2)
        Linear 512 -> 1024
        Bnorm
        LeakyReLU(0.2)
        Linear 1024 -> 768
        Output non-linearity
    """

    def __init__(self, latent_dim):
        super(SimpleGenerator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        x = self.layers(z)
        x = x.view(x.shape[0], 28, 28)
        return x


class SimpleDiscriminator(nn.Module):
    """
    Construct discriminator. You should experiment with your model, but the following is a good start:
        Linear 784 -> 512
        LeakyReLU(0.2)
        Linear 512 -> 256
        LeakyReLU(0.2)
        Linear 256 -> 1
        Output non-linearity
    """

    def __init__(self):
        super(SimpleDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # return discriminator score for img
        x = self.layers(img.view(img.shape[0], 784))
        x = x.squeeze()  # 降维：[batch_size, 1] -> [batch_size]
        return x


class Generator(nn.Module):
    """
    bug: _pickle.PicklingError: Can't pickle <class '__main__.Generator'>: attribute lookup Generator on __main__ failed
    Python中存在原生的Generator类，可能会发生冲突。
    """

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
