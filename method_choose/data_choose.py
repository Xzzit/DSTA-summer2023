from __future__ import print_function, division

from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import shutil
from dataset.dhg_skeleton import DHG_SKE


def init_seed(x):
    # pass
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_choose(args, block):
    if args.data == 'dhg_skeleton':
        workers = args.worker
        data_set_train = DHG_SKE(mode='train', **args.data_param['train_data_param'])
        data_set_val = DHG_SKE(mode='val', **args.data_param['val_data_param'])
    elif args.data == 'shrec_skeleton':
        workers = args.worker
        data_set_train = DHG_SKE(mode='train', **args.data_param['train_data_param'])
        data_set_val = DHG_SKE(mode='train', **args.data_param['val_data_param'])
    else:
        raise (RuntimeError('No data loader'))
    data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                    num_workers=workers, drop_last=False, pin_memory=args.pin_memory,
                                    worker_init_fn=init_seed)
    data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                    num_workers=workers, drop_last=True, pin_memory=args.pin_memory,
                                    worker_init_fn=init_seed)

    block.log('Data load finished: ' + args.data)

    shutil.copy2(__file__, args.model_saved_name)
    return data_loader_train, data_loader_val
