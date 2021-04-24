import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop


class ImageDataset(Dataset):
    """
    This version is for unlabeled problems
    """
    def __init__(self, pt_dataset, d_img, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.d_img = d_img
        self.perm = torch.arange(self.d_img * self.d_img) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = self.d_img * self.d_img - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = (np.array(x) - 127.5) / 127.5
        x = torch.from_numpy(x).view(-1, 3)  # x = x.view(-1, 3)  # flatten out all pixels
        x = x[self.perm].float()  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1)  # cluster assignments
        return a[:-1], a[1:]  # always just predict the next one in the sequence

class ImageDatasetWithLabels(Dataset):
    """
    Labeled dataset
    """
    def __init__(self, pt_dataset, d_img, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.d_img = d_img
        self.perm = torch.arange(self.d_img * self.d_img) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = self.d_img * self.d_img - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = (np.array(x) - 127.5) / 127.5
        x = torch.from_numpy(x).view(-1, 3)  # x = x.view(-1, 3) # flatten out all pixels
        x = x[self.perm].float()  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1)  # cluster assignments
        return a[:-1], y  # predict the labels

def load_dataloaders(traindir, valdir, clusters, batch_size, workers, n_px):  
    """
    Load training and validation data loaders
    """
    # Load training data
    train_trfs = Compose([Resize(256), RandomCrop((224, 224)), Resize((n_px, n_px))])
    train_imgs = ImageFolder(traindir, train_trfs)
    train_data = ImageDatasetWithLabels(train_imgs, n_px, clusters)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, sampler=None)
    print('Training set size:', len(train_imgs))

    # Load validation data
    val_trfs = Compose([Resize(256), CenterCrop((224, 224)), Resize((n_px, n_px))])
    val_imgs = ImageFolder(valdir, val_trfs)
    val_data = ImageDatasetWithLabels(val_imgs, n_px, clusters)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)
    print('Validation set size:', len(val_imgs))

    return train_loader, val_loader

def extract(loader, model, prly, print_freq=100):

    # switch to evaluate mode
    model.eval()

    activations = []
    labels = []

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):

            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            activation = torch.mean(output[1][prly][0], 2)
            activation = activation.view(activation.size(0), -1)
            activations.append(activation)
            labels.append(target)

            if i % print_freq == 0:
                print('Iteration', i, 'of', len(loader))

        activations = torch.cat(activations)
        activations = activations.cpu().numpy()
        print('Cache shape:', activations.shape)

        labels = torch.cat(labels)
        labels = labels.cpu().numpy()
        print('Labels shape:', labels.shape)

    return activations, labels