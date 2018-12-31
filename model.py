import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from torch.autograd import Variable
import numpy as np

args = utils.get_args()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.conv3 = nn.Conv2d(40, 40, 3, 1)
        self.conv4 = nn.Conv2d(40, 80, 3, 1)
        self.conv5 = nn.Conv2d(80, 80, 3, 1)
        self.conv6 = nn.Conv2d(80, 80, 3, 1)
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, y=None, mixup=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if mixup:
            lam = Variable(torch.from_numpy(np.array([np.random.beta(0.5,0.5)]).astype('float32')).cuda())
            x, y = utils.mixup_process(x, y, lam=lam)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        if mixup:
            return x, y
        else:
            return x

