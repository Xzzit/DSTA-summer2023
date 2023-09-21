import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
os.environ['DISPLAY'] = 'localhost:10.0'
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../'])

import setproctitle

from method_choose import *
import parser_args
from utility.log import TimerBlock

from method_choose import *
import parser_args
from utility.log import TimerBlock
from model.dstanet import DSTANet

import torch
from torch import nn
from tqdm import tqdm
from utility.log import IteratorTimer
from collections import OrderedDict


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def rm_module(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def load_model(args, block):

    # Load model
    model = DSTANet(num_class=args.class_num, **args.model_param)
    if args.pre_trained_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pre_trained_model)  # ['state_dict']
        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = rm_module(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys = list(pretrained_dict.keys())
        for key in keys:
            for weight in args.ignore_weights:
                if weight in key:
                    if pretrained_dict.pop(key) is not None:
                        block.log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        block.log('Can Not Remove Weights: {}.'.format(key))
        block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        block.log('Pretrained model load finished: ' + args.pre_trained_model)

    model.cuda()
    model = nn.DataParallel(model, device_ids=args.device_id)
    block.log('copy model to gpu')
    return model


with TimerBlock("Good Luck") as block:
    # params
    args = parser_args.parser_args(block)

    setproctitle.setproctitle(args.model_saved_name)
    block.log('work dir: ' + args.model_saved_name)

    # Init model
    model = load_model(args, block)
    model.eval()

    # Init data loader
    data_loader_train, data_loader_val = data_choose(args, block)

    total_num = 0
    right_num_1 = 0
    right_num_5 = 0

    for data, label in data_loader_val:

        # predict
        data = data.cuda()
        label = label.cuda()
        predict = model(data)

        # calculate acc
        top1 = torch.argmax(predict, dim=1)
        top5 = torch.topk(predict, 5, dim=1)[1]

        num_1 = torch.sum(top1 == label).item()
        num_5 = 0
        for i in range(len(label)):
            num_5 += int(int(label[i]) in top5[i])

        total_num += len(label)
        right_num_1 += num_1
        right_num_5 += num_5

    acc_1 = right_num_1 / total_num
    acc_5 = right_num_5 / total_num
    block.log('Top1 acc: {}'.format(acc_1))
    block.log('Top5 acc: {}'.format(acc_5))