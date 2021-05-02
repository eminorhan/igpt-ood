import os
import argparse
import torch
import numpy as np
from model import load_igpt
from utils import load_dataloaders, extract

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model_size', default='s', type=str, help='Model size', choices=['l', 'm', 's'])
parser.add_argument('--model_path', default='/scratch/eo41/image-gpt/models/s/model.ckpt-1000000.index', type=str, help='model path')
parser.add_argument('--cluster_path', default='/scratch/eo41/image-gpt/models/s/kmeans_centers.npy', type=str, help='clusters path')
parser.add_argument('--train_path', default='/scratch/work/public/imagenet/train', type=str, help='training data path')
parser.add_argument('--val_path', default='/scratch/eo41/imagenet/val', type=str, help='validation path')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--prly', default=5, type=int, help='probe layer')
parser.add_argument('--workers', default=8, type=int, help='number of workers for data loaders')
parser.add_argument('--print_freq', default=100, type=int, help='print results after this many iterations')
parser.add_argument('--partition', default=0, type=int, help='which partition of the data', choices=[0, 1, 2])
parser.add_argument('--fragment', default='trainval', type=str, help='Which part of data to cache', choices=['val', 'train', 'trainval'])

if __name__ == '__main__':
    
    args = parser.parse_args()
    print(args)

    n_px = 32  # The released OpenAI iGPT models were trained with 32x32 images. 
    num_partitions = 3  # number of partitions to cache data into

    assert num_partitions > args.partition, "Partition argument must be smaller than the number of partitions."

    # load model and clusters
    model, clusters = load_igpt(args.model_size, args.model_path, args.cluster_path, n_px)
    model = torch.nn.DataParallel(model).cuda()

    # load train and val loader
    splitted_ind = np.array_split(np.arange(1281167), num_partitions)  # divide into roughly equal parts
    tr_sampler = torch.utils.data.sampler.SubsetRandomSampler(splitted_ind[args.partition])
    train_loader, val_loader = load_dataloaders(args.train_path, args.val_path, clusters, args.batch_size, args.workers, n_px, tr_sampler)

    if args.fragment == 'val':
        val_x, val_y = extract(val_loader, model, args.prly, args.print_freq)
        np.savez('cache_{}_{}_val.npz'.format(args.model_size, args.prly), val_x=val_x, val_y=val_y)
    elif args.fragment == 'train':
        tr_x, tr_y = extract(train_loader, model, args.prly, args.print_freq)
        np.savez('cache_{}_{}_{}_tr.npz'.format(args.model_size, args.prly, args.partition), tr_x=tr_x, tr_y=tr_y)
    elif args.fragment == 'trainval':
        val_x, val_y = extract(val_loader, model, args.prly, args.print_freq)
        tr_x, tr_y = extract(train_loader, model, args.prly, args.print_freq)
        np.savez('cache_{}_{}.npz'.format(args.model_size, args.prly), tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)