from dataloader import create_dataloader
from model import Network
import torch
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt

min_loss = 1e10

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        one_hot_target = utils.to_one_hot(target, 10)    
        output, one_hot_target = model(data, one_hot_target, mixup=True)
        loss = F.binary_cross_entropy_with_logits(output, one_hot_target, reduction='mean') # sum up batch loss
        loss.backward()
        optimizer.step()
        if batch_idx % args['log'].getint('print_interval') == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    tested = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample['image'], sample['label']
            data, target = data.to(device), target.to(device)
            #one_hot_target = utils.to_one_hot(target, 10)    
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            #test_loss += F.binary_cross_entropy_with_logits(output, one_hot_target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            tested += len(target)

    test_loss /= len(test_loader.dataset)
    accuracy = 100*correct/len(test_loader.dataset)
    
    if test_loss < min_loss:
        torch.save(model.state_dict(), args['paths']['save_dir'])
        print('Model Saved')

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

args = utils.get_args()
torch.manual_seed(args['debug'].getint('seed'))
train_loader, test_loader = create_dataloader()

device = torch.device("cuda:0")
model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args['train'].getfloat('learning_rate'),
                             )
if args['paths'].getboolean('load_model_from_save'):
    model.load_state_dict(torch.load(args['paths']['save_dir']))
 
for epoch in range(args['train'].getint('num_epochs')):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)
