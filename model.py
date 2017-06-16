import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

class DenseAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, hidden=300, nb_active=10):
        super().__init__()
        self.nb_active = nb_active
        self.encode = nn.Sequential(
            nn.Linear(w*h*c, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
        )
        self.decode = nn.Sequential(
            nn.Linear(hidden, w*h*c),
            nn.Sigmoid()
        )
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        X = X.view(X.size(0), -1)
        h = self.encode(X)
        h = ksparse(h, nb_active=self.nb_active,inplace=True)
        Xr = self.decode(h)
        Xr = Xr.view(size)
        return Xr



def ksparse(x, nb_active=10, inplace=True):
    for i, xi in enumerate(x.data.tolist()):
        inds = np.argsort(xi)
        inds = inds[::-1]
        inds = inds[nb_active:]
        if len(inds):
            inds = np.array(inds)
            inds = torch.from_numpy(inds).long()
            x.data[i][inds] = 0.
    return x

class ConvAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_filters=64):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(c, nb_filters, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(nb_filters, nb_filters, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(nb_filters, nb_filters, 5, 1, 0),
            nn.ReLU(True),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(nb_filters, c, 13, 1, 0),
            nn.Sigmoid()
        )
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        h = self.encode(X)
        h = spatial_sparsity(h, inplace=True)
        Xr = self.decode(h)
        return Xr
  
def spatial_sparsity(x, inplace=True):
    xf = x.view(x.size(0), x.size(1), -1)
    m, _ = xf.max(2)
    m = m.repeat(1, 1, xf.size(2))
    xf = xf * (xf==m).float()
    xf = xf.view(x.size())
    return xf

def _weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
