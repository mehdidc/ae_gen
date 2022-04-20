import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

class KAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_hidden=300, nb_active=16):
        super().__init__()
        self.nb_hidden = nb_hidden
        self.nb_active = nb_active
        self.encode = nn.Sequential(
            nn.Linear(w*h*c, nb_hidden, bias=False)
        )
        self.bias = nn.Parameter(torch.zeros(w*h*c))
        self.params = nn.ParameterList([self.bias])
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        X = X.view(X.size(0), -1)
        h = self.encode(X)
        Xr, _ = self.decode(h)
        Xr = Xr.view(size)
        return Xr
    
    def decode(self, h):
        thetas, _ = torch.sort(h, dim=1, descending=True)
        thetas = thetas[:, self.nb_active:self.nb_active+1]
        h = h * (h > thetas).float()
        Xr = torch.matmul(h, self.encode[0].weight) + self.bias
        Xr = nn.Sigmoid()(Xr)
        return Xr, h


class ZAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_hidden=300, theta=1):
        super().__init__()
        self.nb_hidden = nb_hidden
        self.theta = theta
        self.encode = nn.Sequential(
            nn.Linear(w*h*c, nb_hidden, bias=False)
        )
        self.bias = nn.Parameter(torch.zeros(w*h*c))
        self.params = nn.ParameterList([self.bias])
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        X = X.view(X.size(0), -1)
        h = self.encode(X)
        Xr, _ = self.decode(h)
        Xr = Xr.view(size)
        return Xr
    
    def decode(self, h):
        h  = h * (h > self.theta).float()
        Xr = torch.matmul(h, self.encode[0].weight) + self.bias
        Xr = nn.Sigmoid()(Xr)
        return Xr, h



class DenseAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, encode_hidden=(300,), decode_hidden=(300,), ksparse=True, nb_active=10, denoise=None):
        super().__init__()
        self.encode_hidden = encode_hidden
        self.decode_hidden = decode_hidden
        self.ksparse = ksparse
        self.nb_active = nb_active
        self.denoise = denoise
        
        # encode layers
        layers = []
        hid_prev = w * h * c
        for hid in encode_hidden:
            layers.extend([
                nn.Linear(hid_prev, hid),
                nn.ReLU(True)
            ])
            hid_prev = hid
        self.encode = nn.Sequential(*layers)
        
        # decode layers
        layers = []
        for hid in decode_hidden:
            layers.extend([
                nn.Linear(hid_prev, hid),
                nn.ReLU(True)
            ])
            hid_prev = hid
        layers.extend([
            nn.Linear(hid_prev, w * h * c),
            nn.Sigmoid()
        ])
        self.decode = nn.Sequential(*layers)
        
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        if self.denoise is not None:
            X = X * Variable((torch.rand(X.size()) <= self.denoise).float()).to(X.device)
        X = X.view(X.size(0), -1)
        h = self.encode(X)
        if self.ksparse:
            h = ksparse(h, nb_active=self.nb_active)
        Xr = self.decode(h)
        Xr = Xr.view(size)
        return Xr



def ksparse(x, nb_active=10):
    mask = torch.ones(x.size())
    for i, xi in enumerate(x.data.tolist()):
        inds = np.argsort(xi)
        inds = inds[::-1]
        inds = inds[nb_active:]
        if len(inds):
            inds = np.array(inds)
            inds = torch.from_numpy(inds).long()
            mask[i][inds] = 0
    return x * Variable(mask).float().to(x.device)


class ConvAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_filters=64, spatial=True, channel=True, channel_stride=4):
        super().__init__()
        self.spatial = spatial
        self.channel = channel
        self.channel_stride = channel_stride
        self.encode = nn.Sequential(
            nn.Conv2d(c, nb_filters, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(nb_filters, nb_filters, 5, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(nb_filters, nb_filters, 5, 1, 0),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(nb_filters, c, 13, 1, 0),
            nn.Sigmoid()
        )
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        h = self.encode(X)
        h = self.sparsify(h)
        Xr = self.decode(h)
        return Xr
    
    def sparsify(self, h):
        if not hasattr(self, 'spatial'):
            self.spatial = True
        if not hasattr(self, 'channel'):
            self.channel = True
        if not hasattr(self, 'channel_stride'):
            self.channel_stride = 4
        if self.spatial:
            h = spatial_sparsity(h)
        if self.channel:
            h = strided_channel_sparsity(h, stride=self.channel_stride)
        return h

class SimpleConvAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_filters=64, spatial=True, channel=True, channel_stride=4):
        super().__init__()
        self.spatial = spatial
        self.channel = channel
        self.channel_stride = channel_stride
        self.encode = nn.Sequential(
            nn.Conv2d(c, nb_filters, 13, 1, 0),
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
        h = self.sparsify(h)
        Xr = self.decode(h)
        return Xr
    
    def sparsify(self, h):
        if self.spatial:
            h = spatial_sparsity(h)
        if self.channel:
            h = strided_channel_sparsity(h, stride=self.channel_stride)
        return h


class DeepConvAE(nn.Module):

    def __init__(self, w=32, h=32, c=1, nb_filters=64, nb_layers=3, spatial=True, channel=True, channel_stride=4):
        super().__init__()
        self.spatial = spatial
        self.channel = channel
        self.channel_stride = channel_stride

        layers = [
            nn.Conv2d(c, nb_filters, 5, 1, 0),
            nn.ReLU(True),
        ]
        for _ in range(nb_layers - 1):
            layers.extend([
                nn.Conv2d(nb_filters, nb_filters, 5, 1, 0),
                nn.ReLU(True),
            ])
        self.encode = nn.Sequential(*layers)
        layers = []
        for _ in range(nb_layers - 1):
            layers.extend([
                nn.ConvTranspose2d(nb_filters, nb_filters, 5, 1, 0),
                nn.ReLU(True),
            ])
        layers.extend([
            nn.ConvTranspose2d(nb_filters, c, 5, 1, 0),
            nn.Sigmoid()
        ])
        self.decode = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, X):
        size = X.size()
        h = self.encode(X)
        h = self.sparsify(h)
        Xr = self.decode(h)
        return Xr
    
    def sparsify(self, h):
        if self.spatial:
            h = spatial_sparsity(h)
        if self.channel:
            h = strided_channel_sparsity(h, stride=self.channel_stride)
        return h


def spatial_sparsity(x):
    xf = x.view(x.size(0), x.size(1), -1)
    m, _ = xf.max(2)
    m = m.view(m.size(0), m.size(1), 1)
    m = m.repeat(1, 1, xf.size(2))
    xf = xf * (xf==m).float()
    xf = xf.view(x.size())
    return xf


def channel_sparsity(x):
    m, _ = x.max(1)
    m = m.repeat(1, x.size(1), 1, 1)
    out = x * (x==m).float()
    return out


def strided_channel_sparsity(x, stride=2):
    if stride == 1:
        return channel_sparsity(x)
    mask = _strided_channel_sparsity_mask(x.data.cpu().numpy(), stride)
    mask = mask.astype('float32')
    mask = torch.from_numpy(mask)
    mask = Variable(mask)
    mask = mask.to(x.device)
    return x * mask


def _strided_channel_sparsity_mask(X, stride):
    B, F = X.shape[0:2]
    w, h = X.shape[2:]
    X_ = X.reshape((B, F, w // stride, stride, h // stride, stride))
    mask = (X_ ==  X_.max(axis=(1, 3, 5), keepdims=True))
    mask = mask.reshape(X.shape)
    return mask


def _weights_init(m):
    if hasattr(m, 'weight'):
        xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
