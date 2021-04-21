import os
import argparse
import torch
from model import load_igpt
from utils import load_dataloaders, freeze_trunk, train, validate

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('data', metavar='DIR', help='path to data')
parser.add_argument('--model_size', default='l', type=str, help='Model size', choices=['l', 'm', 's'])
parser.add_argument('--model_path', default='/scratch/eo41/image-gpt/models/l/model.ckpt-1000000.index', type=str, help='model path')
parser.add_argument('--cluster_path', default='/scratch/eo41/image-gpt/models/l/kmeans_centers.npy', type=str, help='clusters path')
parser.add_argument('--n_classes', default=1000, type=int, help='number of classes in downstream classification task')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=15, type=int, help='epochs')
parser.add_argument('--prly', default=20, type=int, help='probe layer')
parser.add_argument('--workers', default=8, type=int, help='number of workers for data loaders')
parser.add_argument('--print_freq', default=100, type=int, help='print results after this many iterations')

if __name__ == '__main__':
    
    args = parser.parse_args()
    print(args)

    n_px = 32  # The released OpenAI iGPT models were trained with 32x32 images. 

    # load model and clusters
    model, clusters = load_igpt(args.model_size, args.model_path, args.cluster_path, n_px, args.prly, args.n_classes)
    freeze_trunk(model) 

    model = torch.nn.DataParallel(model).cuda()

    # load train and val loader
    train_loader, val_loader = load_dataloaders(args.data, clusters, args.batch_size, args.workers, n_px)

    tr_acc1_list = []
    val_acc1_list = []

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, weight_decay=0.0)

    # training loop
    for epoch in range(1, args.epochs+1):
        tr_acc1 = train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
        tr_acc1_list.append(tr_acc1)

    # validate at end of training
    val_acc1, _, _, _ = validate(val_loader, model)
    val_acc1_list.append(val_acc1)

    torch.save({'tr_acc1_list': tr_acc1_list, 'val_acc1_list': val_acc1_list}, 'testrun.tar')