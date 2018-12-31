from dataloader import create_infer_dataloader
from model import Network
import torch
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
import numpy as np

def infer(args, model, device, infer_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(infer_loader):
            print('{}/{}'.format(batch_idx, len(infer_loader)))
            data = sample['image']
            data = data.to(device)
            #one_hot_target = utils.to_one_hot(target, 10)    
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            predictions += list(pred)
        save_data = np.vstack((np.arange(1, 1+len(predictions)), predictions)).astype(int).T
        np.savetxt(args['paths']['report_path'], save_data, '%d', delimiter=',', header='ImageId,Label', comments='')

            

 

args = utils.get_args()
torch.manual_seed(args['debug'].getint('seed'))
infer_loader = create_infer_dataloader()

device = torch.device("cuda:0")
model = Network().to(device)
model.load_state_dict(torch.load(args['paths']['save_dir']))

infer(args, model, device, infer_loader)
