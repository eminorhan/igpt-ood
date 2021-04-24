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

if __name__ == '__main__':
    
    args = parser.parse_args()
    print(args)

    n_px = 32  # The released OpenAI iGPT models were trained with 32x32 images. 

    # load model and clusters
    model, clusters = load_igpt(args.model_size, args.model_path, args.cluster_path, n_px)
    model = torch.nn.DataParallel(model).cuda()

    # load train and val loader
    train_loader, val_loader = load_dataloaders(args.train_path, args.val_path, clusters, args.batch_size, args.workers, n_px)

    val_x, val_y = extract(val_loader, model, args.prly, args.print_freq)
    tr_x, tr_y = extract(train_loader, model, args.prly, args.print_freq)

    np.savez('cache_{}_{}.npz'.format(args.model_size, args.prly), tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)