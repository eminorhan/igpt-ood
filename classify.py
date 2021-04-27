import os
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description='Train linear classifiers on top of extracted features')
parser.add_argument('--model_size', default='s', type=str, help='Model size', choices=['l', 'm', 's'])
parser.add_argument('--cache_dir', default='/scratch/eo41/image-gpt/caches', type=str, help='directory where we store caches')
parser.add_argument('--prly', default=5, type=int, help='probe layer')
parser.add_argument('--solver', default='saga', type=str, help='solver for sklearn logistic regression', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
parser.add_argument('--max_iter', default=20, type=int, help='max_iter parameter for sklearn logistic regression')

args = parser.parse_args()
print(args)
    
data = np.load(os.path.join(args.cache_dir, 'cache_{}_{}.npz'.format(args.model_size, args.prly)))
tr_x, tr_y, val_x, val_y = data['tr_x'], data['tr_y'], data['val_x'], data['val_y']
print('Loaded data')

clf = LogisticRegression(penalty='none', verbose=1, max_iter=args.max_iter, solver='saga', n_jobs=4).fit(tr_x, tr_y)
print('Fitted logistic regression')

print('Validation accuracy', clf.score(val_x, val_y))
print('Training accuracy', clf.score(tr_x, tr_y))

# Save trained model to file
save_filename = 'logreg_{}_{}.pkl'.format(args.model_size, args.prly)
with open(save_filename, 'wb') as file:
    pickle.dump(clf, file)