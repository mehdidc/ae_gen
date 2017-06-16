import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

import torch.nn as nn
from torch.autograd import Variable
import torch

import torchvision.transforms as transforms
import torchvision.datasets as dset

from model import DenseAE
from model import ConvAE

def save_weights(m):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if w.size(1) == 28*28:
            w = w.view(w.size(0), 1, 28, 28)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('feat.png', gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(0) in (32, 64, 128) and w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('feat.png', gr)

def iterative_refinement(ae, nb_examples=1, nb_iter=10, w=28, h=28, c=1):
    x = torch.rand(nb_iter, nb_examples, c, w, h)
    v = Variable(torch.zeros(nb_examples, c, w, h))
    for i in range(1, nb_iter):
        v.data.copy_(x[i-1])
        x[i] = (ae(v).data).float()
    return x


def main():
    w, h, c = 28, 28, 1
    #ae = DenseAE(w=w, h=h, hidden=500, nb_active=50)
    ae = ConvAE(w=w, h=h, nb_filters=128)
    #ae = torch.load('model.th')
    #optim = torch.optim.Adam(ae.parameters(), lr=0.001)
    optim = torch.optim.Adadelta(ae.parameters(), lr=0.1)
    #optim = torch.optim.SGD(ae.parameters(), lr=0.001, momentum=0.9)
    gamma = 0.99
    dataset = dset.MNIST(
        root='data', 
        download=True,
        transform=transforms.ToTensor()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,
        shuffle=True, 
        num_workers=1
    )
    avg_loss = 0.
    nb_updates = 0
    for epoch in range(10):
        for X, y in dataloader:
            ae.zero_grad()
            X = Variable(X)
            Xr = ae(X)
            loss = ((Xr - X) ** 2).view(X.size(0), -1).sum(1).mean()
            loss.backward()
            optim.step()
            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            if nb_updates % 10 == 0:
                print('Loss : {:.6f}'.format(avg_loss))
                gr = grid_of_images_default(1.0 - np.array(Xr.data.tolist()))
                imsave('rec.png', gr)
                ae.apply(save_weights)
                torch.save(ae, 'model.th')
                g = iterative_refinement(ae, nb_examples=100, nb_iter=30, w=w, h=h, c=c)
                gr = grid_of_images_default(1.0 - np.array(g[-1].tolist()))
                imsave('gen.png', gr)
            nb_updates += 1

main()
