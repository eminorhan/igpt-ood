import os
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import train, validate

parser = argparse.ArgumentParser(description='Train linear classifiers on top of extracted features')
parser.add_argument('--model_size', default='s', type=str, help='Model size', choices=['l', 'm', 's'])
parser.add_argument('--cache_dir', default='/scratch/eo41/image-gpt/caches', type=str, help='directory where we store caches')
parser.add_argument('--prly', default=5, type=int, help='probe layer')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--workers', default=8, type=int, help='data loading workers')
parser.add_argument('--epochs', default=15, type=int, help='number of training epochs')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')

args = parser.parse_args()
print(args)
    
data = np.load(os.path.join(args.cache_dir, 'cache_{}_{}.npz'.format(args.model_size, args.prly)))
tr_x, tr_y, val_x, val_y = data['tr_x'], data['tr_y'], data['val_x'], data['val_y']
print('Loaded data')

# training dataset and loader
tr_x = torch.from_numpy(tr_x)
tr_y = torch.from_numpy(tr_y)
tr_dataset = torch.utils.data.TensorDataset(tr_x, tr_y)
tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)

# validation dataset and loader
val_x = torch.from_numpy(val_x)
val_y = torch.from_numpy(val_y)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=None)
print('Prepared data loaders')

# model, loss, optimizer
model = torch.nn.Linear(tr_x.shape[-1], 1000)
model = torch.nn.DataParallel(model).cuda()
print(model)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.0005, weight_decay=0.0)

tr_acc1_list = []

print('Starting training')
for epoch in range(args.epochs):
    # train for one epoch
    acc1 = train(tr_loader, model, criterion, optimizer, epoch, args)
    tr_acc1_list.append(acc1)

# validate at end of epoch
val_acc1 = validate(val_loader, model, args)

savefile_name = 'linclass_{}_{}.tar'.format(args.model_size, args.prly)
torch.save({'tr_acc1_list': tr_acc1_list,
            'val_acc1': val_acc1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, savefile_name)