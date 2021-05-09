import os
import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import AverageMeter, accuracy
from constants import indices_in_1k

parser = argparse.ArgumentParser(description='Train linear classifiers on top of extracted features')
parser.add_argument('--cache_dir', default='/scratch/eo41/image-gpt/caches', type=str, help='directory where we store caches')
parser.add_argument('--classifier_dir', default='/scratch/eo41/image-gpt/classifiers', type=str, help='directory where we store trained classifiers')
parser.add_argument('--eval_data', default='ina', type=str, help='which data to evaluate on', choices=['ina', 'ins', 'inp', 'inc', 'inr'])
parser.add_argument('--model_size', default='s', type=str, help='Model size', choices=['l', 'm', 's'])
parser.add_argument('--prly', default=12, type=int, help='probe layer')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--workers', default=4, type=int, help='data loading workers')

args = parser.parse_args()
print(args)
    
data = np.load(os.path.join(args.cache_dir, args.eval_data, 'cache_{}_{}.npz'.format(args.model_size, args.prly)))
val_x, val_y = data['val_x'], data['val_y']
print('Loaded evaluation data')

# evaluation dataset and loader
val_x = torch.from_numpy(val_x)
val_y = torch.from_numpy(val_y)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=None)
print('Prepared data loaders')

# load model
model = torch.nn.Linear(val_x.shape[-1], 1000)
model = torch.nn.DataParallel(model).cuda()

classifier_path = os.path.join(args.classifier_dir, 'linclass_{}_{}.tar'.format(args.model_size, args.prly))
if os.path.isfile(classifier_path):
    print("=> loading trained classifier at '{}'".format(classifier_path))
    checkpoint = torch.load(classifier_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("=> no trained classifier found at '{}'".format(classifier_path))
    # TODO: break here
print(model)

print('Starting evaluation')
preds = []
targets = []

# switch to eval mode
model.eval()

with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda()
        target = target.cuda()

        # compute output
        if args.eval_data == 'ina':
            output = model(images)[:, indices_in_1k]
        else:
            output = model(images)
        print(output.shape)    

        preds.append(np.argmax(output.cpu().numpy(), axis=1))
        targets.append(target.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    print(preds.shape)
    print(targets.shape)

    assert preds.shape == targets.shape, 'prediction shape should match target shape.'

    acc1 = np.mean(preds == targets)
    print('Top-1 accuracy on evaluation data:', acc1)    

