# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 06:16:43 2023

@author: sushant
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """
        Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class LinearFF(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(device)]
            
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print("Training layer", i, '....')
            h_pos, h_neg = layer.train(h_pos, h_neg)
            

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # this is basically the direction of the input,
        # x_hat
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # matmul with weights, forward pass
        return self.relu(
            torch.mm(x_direction, self.weight.T)+
            self.bias.unsqueeze(0)
            )        
        
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            # we want to increase goodness for g_pos
            # so we negate it and add our threshold
            # the optimizer will try to reduce -g_pos
            # which will increase g_pos or goodness of pos input
            loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))

            # we want to decrease goodness for g_neg
            # so we subtract the threshold
            # optimizer will try to reduce g_neg
            # which will decrease goodness of neg input
            loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))

            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            # finally our loss is sum of both
            loss = loss_pos.mean() + loss_neg.mean()

            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()

        # done with epochs, return outputs
        # detach because we dont want to gradients to flow back
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    
    
class FFMLP(nn.Module):
    def __init__(self, device):
        super(FFMLP, self).__init__()

        self.layers = [
            LinearFF(in_features=784, out_features=300).to(device),
            LinearFF(in_features=300, out_features=300).to(device),
            LinearFF(in_features=300, out_features=300).to(device),
        ]

    def train(self, x_pos, x_neg, num_epochs):
        x_pos_hat, x_neg_hat = x_pos, x_neg
        for layer in self.layers:
            x_pos_hat, x_neg_hat = layer.train(x_pos_hat, x_neg_hat, num_epochs)

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    

   
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = LinearFF([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())