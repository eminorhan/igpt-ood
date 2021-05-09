import numpy as np

model_size = 'm'
layers = [15, 16, 17, 18, 19, 20]

for l in layers:
    val_data = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_val.npz'.format(model_size, l))
    val_x, val_y = val_data['val_x'], val_data['val_y']

    tr_data = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_0_tr.npz'.format(model_size, l))
    tr_x, tr_y = tr_data['tr_x'], tr_data['tr_y']

    print('tr_x shape', tr_x.shape)
    print('tr_y shape', tr_y.shape)

    np.savez('cache_{}_{}.npz'.format(model_size, l), tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)
    print('Saved size', model_size, 'layer', l)

model_size = 'l'
layers = [20, 21, 22, 23, 24, 25]

for l in layers:
    val_data = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_val.npz'.format(model_size, l))
    val_x, val_y = val_data['val_x'], val_data['val_y']

    tr_data_0 = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_0_tr.npz'.format(model_size, l))
    tr_x_0, tr_y_0 = tr_data_0['tr_x'], tr_data_0['tr_y']

    tr_data_1 = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_1_tr.npz'.format(model_size, l))
    tr_x_1, tr_y_1 = tr_data_1['tr_x'], tr_data_1['tr_y']

    tr_data_2 = np.load('/scratch/eo41/image-gpt/caches/cache_{}_{}_2_tr.npz'.format(model_size, l))
    tr_x_2, tr_y_2 = tr_data_2['tr_x'], tr_data_2['tr_y']

    tr_x = np.concatenate((tr_x_0, tr_x_1, tr_x_2))
    tr_y = np.concatenate((tr_y_0, tr_y_1, tr_y_2))

    print('tr_x shape', tr_x.shape)
    print('tr_y shape', tr_y.shape)

    np.savez('cache_{}_{}.npz'.format(model_size, l), tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)
    print('Saved size', model_size, 'layer', l)
