import os
import time
import json
import hashlib
import argparse
import time

import torch
import torch.utils.data
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torch import autograd
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.datasets

# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Using ', device)

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])

def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader



class VAE(nn.Module):
    def __init__(self, batch_size, L):
        super(VAE, self).__init__()
        self.L = L
        self.bs = batch_size

        self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3),
                    nn.AvgPool2d(kernel_size=2, stride=2 ),
                    nn.ELU(),
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.ELU(),
                    nn.Conv2d(64, 256, kernel_size=5, stride=2),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.ELU()
                    )
        self.params = nn.Linear(256, self.L*2)
        self.linear = nn.Linear(self.L, 256)
        self.decoder = nn.Sequential(
                    nn.ELU(),
                    nn.Conv2d(256, 64, kernel_size=5, padding=4),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.ELU(),
                    nn.Conv2d(64, 32, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.ELU(),
                    nn.Conv2d(32, 16, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.Conv2d(16, 3, kernel_size=3, padding=2)
                    )

    def encode(self, x):
        x = self.encoder(x)
        return self.params(x.view(-1, 256))

    def reparameterize(self, q_params):
        mu, log_sigma = q_params[:,:self.L], q_params[:,self.L:]
        # print('mu ' , mu.size())
        # print('log_sigma ' , log_sigma.size())
        sigma = torch.exp(log_sigma) + 1e-7
        # print('sigma ' , sigma.size())

        e = torch.randn(self.bs, self.L, device=device)
        # print('e ' , e.size())
        z = mu + sigma * e
        return z, mu, log_sigma

    def decode(self, z):
        z = self.linear(z)
        return self.decoder(z.view(-1, 256, 1, 1))

    def forward(self, x):
        q_params = self.encode(x)
        z, mu, log_sigma = self.reparameterize(q_params)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma


def ELBO(x, recon_x, mu, log_sigma):
    """
    Function that computes the negative ELBO
    """
    # Compute KL Divergence
    kl = 0.5 * (-1. - log_sigma + torch.exp(log_sigma)**2. + mu**2.).sum(dim=1)
    # Compute reconstruction error
    logpx_z = F.binary_cross_entropy_with_logits(x.view(-1, 784), recon_x.view(-1, 784)).sum(dim=-1)
    return -(logpx_z - kl).mean()

def KL(x, recon_x, mu, log_sigma):
    """
    Function that computes the negative ELBO
    """
    # Compute KL Divergence
    kl = 0.5 * (-1. - log_sigma + torch.exp(log_sigma)**2. + mu**2.).sum(dim=1)
    
    return kl

if __name__ == "__main__":
    batch_size = 32
    # Load dataset
    print("Loading datasets.....")
    start_time = time.time()
    train, valid, test = get_data_loader("svhn", batch_size)
    print("DONE in {:.2f} sec".format(time.time() - start_time))

    # Set hyperparameters
    model = VAE(batch_size=batch_size, L=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    n_epochs=20

    dataloader = {"Train": train,
                  "Valid": valid}

    loss_func = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        epoch += 1
        print("Epoch {} of {}".format(epoch, n_epochs))
        accuracy = {}

        for loader in ["Train", "Valid"]:
            start_time = time.time()
            train_epoch_loss = 0
            valid_epoch_loss = 0
            counter = 0

            if loader == "Train":
                model.train()
            else:
                model.eval()

            for i, (x, y) in enumerate(dataloader[loader]):
                x.to(device)
                y.to(device)
                print(x.size())
                counter += 1
                optimizer.zero_grad()
                # forward pass
                recon_x, mu, log_sigma = model(x)
                print(x.shape)
                print(recon_x.shape)
                loss = loss_func(recon_x, x) + KL(x, recon_x, mu, log_sigma)

                if loader != "Valid":
                    loss.backward()
                    optimizer.step()
                    train_epoch_loss += loss.item()
                else:
                    valid_epoch_loss += loss.item()

            if loader != "Valid":
                print("Train Epoch loss: {:.6f}".format(-train_epoch_loss / counter))
            else:
                print("Valid Epoch loss: {:.6f}".format(-valid_epoch_loss / counter))