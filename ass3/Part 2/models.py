import torch.nn as nn


class SimpleGenerator(nn.Module):
    """
    bug: _pickle.PicklingError: Can't pickle <class '__main__.Generator'>: attribute lookup Generator on __main__ failed
    Python中存在原生的Generator类，可能会发生冲突。
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
