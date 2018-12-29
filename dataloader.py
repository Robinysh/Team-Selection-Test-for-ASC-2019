import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import numpy as np
import os
import utils
import torchvision.transforms as transforms


args = utils.get_args()

class DigitsDataSet(Dataset):
    def __init__(self, csv_file, pre_transform=None, post_transform=None, labeled=True):
        if os.path.exists(args['paths']['processed_file'])\
           and args['data'].getboolean('load_data_from_save'):
            data = np.load(args['paths']['processed_file'])
        else:
            data = np.genfromtxt(csv_file, skip_header=1, delimiter=',', dtype=np.uint8)
            np.save(args['paths']['processed_file'], data)

        #self.images = torch.from_numpy(np.reshape(data[:,1:], (-1, 1, 28, 28))).float()
        #reshape image from Nx(HxW) order to NxHxWxC order
        self.images = np.reshape(data[:,1:], (-1, 28, 28, 1))
        if labeled:
            self.labels = torch.from_numpy(data[:,0]).long()

        if pre_transform:
            self.images = pre_transform(self.images)
        self.labeled = labeled 
        self.transform = post_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labeled:
            sample = {'image': self.images[idx], 'label': self.labels[idx]}
        else:
            sample = {'image': self.images[idx]}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample      

        
def create_dataloader():
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    #create master dataset
    master = DigitsDataSet(csv_file=args['paths']['train_file_name'], post_transform=transform)

    #split the dataset into train and test
    n_test = int(len(master)*args['data'].getfloat('test_percentage'))
    n_train = len(master) - n_test
    train_dataset, test_dataset = data.random_split(master, (n_train, n_test))

    #create dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args['data'].getint('batch_size'),
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args['data'].getint('batch_size'),
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 )
    return train_dataloader, test_dataloader

