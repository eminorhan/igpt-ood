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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_dataloaders(traindir, valdir, clusters, batch_size, workers, n_px, tr_sampler=None):  
    """
    Load training and validation data loaders
    """
    # Load training data
    train_trfs = Compose([Resize(256), RandomCrop((224, 224)), Resize((n_px, n_px))])
    train_imgs = ImageFolder(traindir, train_trfs)
    train_data = ImageDatasetWithLabels(train_imgs, n_px, clusters)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=tr_sampler)
    print('Training set size:', len(train_imgs))
    print('Training loader size:', len(train_loader))

    # Load validation data
    val_trfs = Compose([Resize(256), CenterCrop((224, 224)), Resize((n_px, n_px))])
    val_imgs = ImageFolder(valdir, val_trfs)
    val_data = ImageDatasetWithLabels(val_imgs, n_px, clusters)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)
    print('Validation set size:', len(val_imgs))
    print('Validation loader size:', len(val_loader))

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

            activation = torch.mean(output[1][prly][1], 2)
            activation = activation.view(activation.size(0), -1)
            activations.append(activation.cpu().numpy())
            labels.append(target.cpu().numpy())

            if i % print_freq == 0:
                print('Iteration', i, 'of', len(loader))

        activations = np.concatenate(activations)
        print('Cache shape:', activations.shape)

        labels = np.concatenate(labels)
        print('Labels shape:', labels.shape)

    return activations, labels

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

def inst_accuracies(labels, preds):
    from constants import conversion_table

    shape_matches = np.zeros(len(preds))
    texture_matches = np.zeros(len(preds))

    for i in range(len(preds)):
        pred = preds[i]
        texture = labels['textures'][i]
        shape = labels['shapes'][i]

        if pred in conversion_table[texture]:
            texture_matches[i] = 1

        if pred in conversion_table[shape]:
            shape_matches[i] = 1

    correct = shape_matches + texture_matches  # texture or shape predictions
    frac_correct = np.mean(correct)
    frac_shape = np.sum(shape_matches) / np.sum(correct)
    frac_texture = np.sum(texture_matches) / np.sum(correct)

    return frac_correct, frac_shape, frac_texture

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()

def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            preds = np.argmax(output.cpu().numpy(), axis=1)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg