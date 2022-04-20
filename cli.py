import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from clize import run
import numpy as np
from skimage.io import imsave

from viz import grid_of_images_default

import torch.nn as nn
import torch

from model import DenseAE
from model import ConvAE
from model import DeepConvAE
from model import SimpleConvAE
from model import ZAE
from model import KAE
from data import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_dataset(code_2d, categories):
    colors = [
        'r',
        'b',
        'g',
        'crimson',
        'gold',
        'yellow',
        'maroon',
        'm',
        'c',
        'orange'
    ]
    for cat in range(0, 10):
        g = (categories == cat)
        plt.scatter(
            code_2d[g, 0], 
            code_2d[g, 1],
            marker='+', 
            c=colors[cat], 
            s=40, 
            alpha=0.7,
            label="digit {}".format(cat)
        )


def plot_generated(code_2d, categories):
    g = (categories < 0)
    plt.scatter(
        code_2d[g, 0], 
        code_2d[g, 1], 
        marker='+',
        c='gray', 
        s=30
    )


def grid_embedding(h):
    from lapjv import lapjv
    from scipy.spatial.distance import cdist
    assert int(np.sqrt(h.shape[0])) ** 2 == h.shape[0], 'Nb of examples must be a square number'
    size = int(np.sqrt(h.shape[0]))
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
    cost_matrix = cdist(grid, h, "sqeuclidean").astype('float32')
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    _, rows, cols = lapjv(cost_matrix)
    return rows


def save_weights(m, folder='.'):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if w.size(1) == 28*28 or w.size(0) == 28*28:
            w0, w1 = w.size(0), w.size(1)
            if w0 == 28*28:
                w = w.transpose(0, 1)
                w = w.contiguous()
            w = w.view(w.size(0), 1, 28, 28)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/feat_{}.png'.format(folder, w0), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(0) in (32, 64, 128, 256, 512) and w.size(1) in (1, 3):
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/feat.png'.format(folder), gr)

@torch.no_grad()
def iterative_refinement(ae, nb_examples=1, nb_iter=10, w=28, h=28, c=1, batch_size=None):
    if batch_size is None:
        batch_size = nb_examples
    x = torch.rand(nb_iter, nb_examples, c, w, h)
    for i in range(1, nb_iter):
        for j in range(0, nb_examples, batch_size):
            oldv = x[i-1][j:j + batch_size].to(device)
            newv = ae(oldv)
            newv = newv.data.cpu()
            x[i][j:j + batch_size] = newv
    return x


def build_model(name, w, h, c):
    if name == 'convae':
        ae = ConvAE(
            w=w, h=h, c=c, 
            nb_filters=128, 
            spatial=True, 
            channel=True, 
            channel_stride=4,
        )
    elif name == 'zae':
        ae = ZAE(
            w=w, h=h, c=c,
            theta=3,
            nb_hidden=1000,
        )
    elif name == 'kae':
        ae = KAE(
            w=w, h=h, c=c,
            nb_active=1000,
            nb_hidden=1000,
        )
    elif name == 'denseae':
        ae = DenseAE(
            w=w, h=h, c=c,
            encode_hidden=[1000],
            decode_hidden=[],
            ksparse=True,
            nb_active=50,
        )
    elif name == 'simple_convae':
        ae = SimpleConvAE(
            w=w, h=h, c=c,
            nb_filters=128,
        )
    elif name == 'deep_convae':
        ae = DeepConvAE(
            w=w, h=h, c=c, 
            nb_filters=128, 
            spatial=True, 
            channel=True, 
            channel_stride=4,
            nb_layers=3, 
        )
    else:
        raise ValueError('Unknown model')

    return ae


def salt_and_pepper(X, proba=0.5):
    a = (torch.rand(X.size()).to(device) <= (1 - proba)).float()
    b = (torch.rand(X.size()).to(device) <= 0.5).float()
    c = ((a == 0).float() * b)
    return X * a + c


def train(*, dataset='mnist', folder='mnist', resume=False, model='convae', walkback=False, denoise=False, epochs=100, batch_size=64, log_interval=100):
    gamma = 0.99
    dataset = load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    c, h, w = x0.size()
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4
    )
    if resume:
        ae = torch.load('{}/model.th'.format(folder))
        ae = ae.to(device)
    else:
        ae = build_model(model, w=w, h=h, c=c)
        ae = ae.to(device)
    optim = torch.optim.Adadelta(ae.parameters(), lr=0.1, eps=1e-7, rho=0.95, weight_decay=0)
    avg_loss = 0.
    nb_updates = 0
    _save_weights = partial(save_weights, folder=folder)

    for epoch in range(epochs):
        for X, y in dataloader:
            ae.zero_grad()
            X = X.to(device)
            if hasattr(ae, 'nb_active'):
                ae.nb_active = max(ae.nb_active - 1, 32)
            # walkback + denoise
            if walkback:
                loss = 0.
                x = X.data
                nb = 5
                for _ in range(nb):
                    x = salt_and_pepper(x, proba=0.3) # denoise
                    x = x.to(device)
                    x = ae(x) # reconstruct
                    Xr = x
                    loss += (((x - X) ** 2).view(X.size(0), -1).sum(1).mean()) / nb
                    x = (torch.rand(x.size()).to(device) <= x.data).float() # sample
            # denoise only
            elif denoise:
                Xc = salt_and_pepper(X.data, proba=0.3)
                Xr = ae(Xc)
                loss = ((Xr - X) ** 2).view(X.size(0), -1).sum(1).mean()
            # normal training
            else:
                Xr = ae(X)
                loss = ((Xr - X) ** 2).view(X.size(0), -1).sum(1).mean()
            loss.backward()
            optim.step()
            avg_loss = avg_loss * gamma + loss.item() * (1 - gamma)
            if nb_updates % log_interval == 0:
                print('Epoch : {:05d} AvgTrainLoss: {:.6f}, Batch Loss : {:.6f}'.format(epoch, avg_loss, loss.item()  ))
                gr = grid_of_images_default(np.array(Xr.data.tolist()))
                imsave('{}/rec.png'.format(folder), gr)
                ae.apply(_save_weights)
                torch.save(ae, '{}/model.th'.format(folder))
            nb_updates += 1


def test(*, dataset='mnist', folder='mnist', nb_iter=100, nb_generate=100, tsne=False):
    dataset = load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    c, h, w = x0.size()
    nb = nb_generate
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=nb,
        shuffle=True, 
        num_workers=1
    )
    print('Load model...')
    ae = torch.load('{}/model.th'.format(folder), map_location="cpu")
    ae = ae.to(device)
    def enc(X):
        batch_size = 64
        h_list = []
        for i in range(0, X.size(0), batch_size):
            x = X[i:i + batch_size]
            x = x.to(device)
            name = ae.__class__.__name__
            if name in ('ConvAE',):
                h = ae.encode(x)
                h, _ = h.max(2)
                h = h.view((h.size(0), -1))
            elif name in ('DenseAE',):
                x = x.view(x.size(0), -1)
                h = x
                #h = ae.encode(x)
            else:
                h = x.view(x.size(0), -1)
            h = h.data.cpu()
            h_list.append(h)
        return torch.cat(h_list, 0)

    print('iterative refinement...')
    g = iterative_refinement(
        ae, 
        nb_iter=nb_iter, 
        nb_examples=nb, 
        w=w, h=h, c=c, 
        batch_size=64
    )
    np.savez('{}/generated.npz'.format(folder), X=g.numpy())
    g_subset = g[:, 0:100]
    gr = grid_of_images_default(g_subset.reshape((g_subset.shape[0]*g_subset.shape[1], h, w, 1)).numpy(), shape=(g_subset.shape[0], g_subset.shape[1])) 
    imsave('{}/gen_full_iters.png'.format(folder), gr)

    g = g[-1] # last iter
    print(g.shape)
    gr = grid_of_images_default(g.numpy())
    imsave('{}/gen_full.png'.format(folder), gr)

    if tsne:
        from sklearn.manifold import TSNE
        print('Load data...')
        X, y = next(iter(dataloader))
        print('Encode data...')
        xh = enc(X)
        print('Encode generated...')
        gh = enc(g)
        X = X.numpy()
        g = g.numpy()
        xh = xh.numpy()
        gh = gh.numpy()

        a = np.concatenate((X, g), axis=0)
        ah = np.concatenate((xh, gh), axis=0)
        labels = np.array(y.tolist() + [-1] * len(g))
        sne = TSNE()
        print('fit tsne...')
        ah = sne.fit_transform(ah)
        print('grid embedding...')
        
        asmall = np.concatenate((a[0:450], a[nb:nb + 450]), axis=0)
        ahsmall = np.concatenate((ah[0:450], ah[nb:nb + 450]), axis=0)
        rows = grid_embedding(ahsmall)
        asmall = asmall[rows]
        gr = grid_of_images_default(asmall)
        imsave('{}/sne_grid.png'.format(folder), gr)

        fig = plt.figure(figsize=(10, 10))
        plot_dataset(ah, labels)
        plot_generated(ah, labels)
        plt.legend(loc='best')
        plt.axis('off')
        plt.savefig('{}/sne.png'.format(folder))
        plt.close(fig)



if __name__ == '__main__':
    run([train, test])
