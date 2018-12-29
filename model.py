import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #self.conv1 = nn.Conv2d(1, 20, 3, 1)
        #self.conv2 = nn.Conv2d(20, 40, 3, 1)
        self.fc1 = nn.Linear(784, 40)
        self.fc2 = nn.Linear(40, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

