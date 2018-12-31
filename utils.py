import configparser
import os
import numpy as np
import torch
from torch.autograd import Variable

def get_args():
    config = configparser.ConfigParser()
    if not os.path.exists('config.ini'):
        raise IOError('config.ini not found.')
    config.read('config.ini')
    return config

def normalize(images):
    return (images-images.mean(axis=0))/images.std(axis=0)

def mixup_process(out, target_reweighted,lam):

    indices = np.random.permutation(out.size(0))
    out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
    return out, target_reweighted

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)

