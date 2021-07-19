import os
import torch
from glob import glob


def resumeCheckpoint(net, hyperparameters):
    # Load checkpoint.
    #print('==> Resuming from checkpoint..')

    checkpointPath = './experiences/' + hyperparameters['name'] + '/checkpoint'
    assert os.path.isdir(checkpointPath), 'Error: no checkpoint directory found!'
    path = sorted(glob(checkpointPath + '/*.pt7'), key=os.path.getmtime)[-1]
    print(path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return checkpoint['best_loss'], checkpoint['epoch'], net