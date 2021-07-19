import numpy as np
import visdom
import sys
sys.path.insert(0, '../')
from helper_functions.loadConfig import loadConfig
from datasets.CLSA_dataloader import CLSA_dataloader_train, CLSA_dataloader_valid
from sklearn.metrics import r2_score
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
from models.metaEfficient import get_metaEfficient


def trainer(lower, upper, var_n):
    global expname
    # =========  Load settings from Config file ==========
    hyperparameters = loadConfig('./config.txt')

    # ========== Define parameters here ==========
    device = hyperparameters['device']
    total_epoch = hyperparameters['N_epochs']
    path_dataset = hyperparameters['path_dataset']
    bs =hyperparameters['batch_size']
    num_classes = 1
    lr = hyperparameters['lr']

    # ========== Define dataloader here ==========
    train_set = CLSA_dataloader_train(path_dataset, var_n, lower, hyperparameters=hyperparameters)
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=8, shuffle=True)
    val_set = CLSA_dataloader_valid(path_dataset, var_n, lower, upper, hyperparameters=hyperparameters)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=8)

    # ========== Define model here ==========
    # Attention
    # from models.attention_model import AttentionNet
    # net = AttentionNet(n_output=num_classes)
    # net = nn.Sequential(nn.AdaptiveAvgPool2d((587, 587)), net)
    # model_name = "AttentionNet"

    # Meta EfficientNet
    # net = get_metaEfficient(n_features=301)
    # model_name = "MetaEfficientNetB3"

    # Meta
    # net = nn.Sequential(nn.Linear(301, 500),
    #                           nn.BatchNorm1d(500),
    #                           nn.ReLU(),
    #                           nn.Dropout(p=0.2),
    #                           nn.Linear(500, 250),  # FC layer output will have 250 features
    #                           nn.BatchNorm1d(250),
    #                           nn.ReLU(),
    #                           nn.Dropout(p=0.2),
    #                           nn.Linear(250, 1),
    #                           nn.ReLU())
    # model_name = "MetaNet"

    # EfficientNet
    net = EfficientNet.from_pretrained('efficientnet-b3', in_channels=3)
    net = nn.Sequential(nn.AdaptiveAvgPool2d((300, 300)), net, nn.Linear(1000, 500), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(500, num_classes), MemoryEfficientSwish())
    model_name = "EfficientNetB3"

    # Set all grads to true
    for param in net.parameters():
        param.requires_grad = True

    if hyperparameters['device'] == 'cuda':
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # =========  Criterion, optimizer, scheduler ============
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # For regression
    criterion = nn.L1Loss()
    # For classification
    # criterion = nn.BCELoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters["N_epochs"], eta_min=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    # =========  Start visdom to visualize training ==========
    expName = hyperparameters["name"] + "_" + model_name + "_" + str(lr) + "_" + str(bs) + "_" + path_dataset[2:-5]
    if not os.path.isdir('experiences/' + expName):
        os.mkdir('experiences/' + expName)
    viz = visdom.Visdom(env=expName)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    # =========  Main  ============
    print("Training")

    global val_win, train_win, test_win_0, test_win_1, w_win, best_acc, best_loss
    val_win, train_win, test_win_0, test_win_1, w_win = None, None, None, None, None
    best_acc = 0
    best_loss = 1000000000000000000
    for epoch in range(0, total_epoch):
        try:
            print("Learning rate = %4f\n" % scheduler._last_lr[0])
        except:
            print("Learning rate = %4f\n" % lr)
        train(epoch, optimizer, criterion, scheduler, hyperparameters, train_loader, device, viz, net)
        val_loss = valid(epoch, criterion, val_loader, device, viz, net, expName)
        scheduler.step()


# ========= Train ============
def train(epoch, optimizer, criterion, scheduler, hyperparameters, train_loader, device, viz, net):
    global val_win, train_win, test_win_0, test_win_1
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    targ_array_0 = []
    pred_array_0 = []

    for batch_idx, (inputs, targets, meta) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        inputs, targets, meta = inputs.to(device), targets.to(device), meta.to(device)
        # inputs = [inputs, meta]
        # inputs = meta
        outputs = net(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        targ_array_0.extend(targets.detach().cpu().numpy())
        pred_array_0.extend(outputs.detach().cpu().numpy())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # roc = roc_auc_score(np.array(targ_array_0).argmax(axis=1), np.array(pred_array_0).argmax(axis=1))
    mean_losses_train = train_loss / np.float32(len(train_loader))

    try:
        r2_0 = r2_score(np.array(targ_array_0), np.array(pred_array_0))
    except:
        print("Can't compute R2 train")
        r2_0 = 0
    if r2_0 < -1:
        r2_0 = 0

    update = None if val_win is None else 'append'
    val_win = viz.line(
        X=np.expand_dims(np.array(epoch), axis=0),
        Y=np.expand_dims(np.array(mean_losses_train), axis=0),
        win=val_win,
        update=update,
        name="Training",
        opts={'title': "Loss", 'xlabel': "Iterations", 'ylabel': "Loss"}
    )
    update = None if test_win_0 is None else 'append'
    test_win_0 = viz.line(
        X=np.expand_dims(np.array(epoch), axis=0),
        Y=np.expand_dims(np.array(r2_0), axis=0),
        win=test_win_0,
        update=update,
        name="Training",
        opts={'title': "R2", 'xlabel': "Iterations", 'ylabel': "R2"}
    )
    print("Epoch %d: Train Loss %4f\n" % (epoch, mean_losses_train))


# ========= Test ============
def valid(epoch, criterion, val_loader, device, viz, net, expName):
    global best_acc, val_win, test_win_0, test_win_1, best_loss
    net.eval()
    test_loss = 0
    targ_array_0 = []
    pred_array_0 = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, meta) in enumerate(val_loader):
            inputs, targets, meta = inputs.to(device), targets.to(device), meta.to(device)
            # inputs = [inputs, meta]
            # inputs = meta
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            test_loss += loss.item()
            targ_array_0.extend(targets.detach().cpu().numpy())  # [0])
            pred_array_0.extend(outputs.detach().cpu().numpy())  # [0])


        # roc = roc_auc_score(np.array(targ_array_0).argmax(axis=1), np.array(pred_array_0).argmax(axis=1))
        mean_losses_valid = test_loss / np.float32(len(val_loader))
        try:
            r2_0 = r2_score(np.array(targ_array_0), np.array(pred_array_0))
        except:
            print("Can't compute R2 val")
            r2_0 = 0
        if r2_0 < -1:
            r2_0 = 0

        val_win = viz.line(
            X=np.expand_dims(np.array(epoch), axis=0),
            Y=np.expand_dims(np.array(mean_losses_valid), axis=0),
            win=val_win,
            update='append',
            name="Validation",
            opts={'title': "Loss", 'xlabel': "Iterations", 'ylabel': "Loss"}
        )
        test_win_0 = viz.line(
            X=np.expand_dims(np.array(epoch), axis=0),
            Y=np.expand_dims(np.array(r2_0), axis=0),
            win=test_win_0,
            update='append',
            name="Validation",
            opts={'title': "R2_age", 'xlabel': "Iterations", 'ylabel': "R2"}
        )

        print('Valid Loss: {:.4f}'.format(mean_losses_valid))

    # Save checkpoint.
    if mean_losses_valid < best_loss:
        best_loss = mean_losses_valid
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('experiences/' + expName + '/checkpoint'):
            os.mkdir('experiences/' + expName + '/checkpoint')
        torch.save(state, './experiences/' + expName + '/checkpoint/epoch_' + str(epoch) + '_best_loss_'
                   + str(best_loss)[0:5] + '.pt7')
    return mean_losses_valid

if __name__ == '__main__':
    var_n = 28
    lower = 18000
    upper = 21860
    trainer(lower, upper, var_n,)
