import torch

import torchvision.transforms as transforms
import torchvision.datasets as dset


class Invert:
    def __call__(self, x):
        return 1 - x

class Gray:
    def __call__(self, x):
        return x[0:1]



def load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root='data/mnist', 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        return dataset
    elif dataset_name == 'coco':
        dataset = dset.ImageFolder(root='data/coco',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'quickdraw':
        X = (np.load('data/quickdraw/teapot.npy'))
        X = X.reshape((X.shape[0], 28, 28))
        X  = X / 255.
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        dataset = TensorDataset(X, X)
        return dataset
    elif dataset_name == 'shoes':
        dataset = dset.ImageFolder(root='data/shoes/ut-zap50k-images/Shoes',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'footwear':
        dataset = dset.ImageFolder(root='data/shoes/ut-zap50k-images',
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'celeba':
        dataset = dset.ImageFolder(root='data/celeba',
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'birds':
        dataset = dset.ImageFolder(root='data/birds/'+split,
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
         ]))
        return dataset
    elif dataset_name == 'sketchy':
        dataset = dset.ImageFolder(root='data/sketchy/'+split,
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            Gray()
         ]))
        return dataset
 
    elif dataset_name == 'fonts':
        dataset = dset.ImageFolder(root='data/fonts/'+split,
            transform=transforms.Compose([
            transforms.ToTensor(),
            Invert(),
            Gray(),
         ]))
        return dataset
    else:
        raise ValueError('Error : unknown dataset')
