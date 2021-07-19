###################################################
#
#   Script to
#   - Calculate prediction of the test datasets
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import torch
import os
import h5py
import cv2 as cv
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import time
from scipy import stats
import random
import cv2 as cv
import matplotlib.cm as cm
import copy
from PIL import Image

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# help_functions.py
from helper_functions.loadConfig import loadConfig
from helper_functions.resumeCheckpoint import resumeCheckpoint
from datasets.CLSA_dataloader import CLSA_dataloader_test
from sklearn.metrics import r2_score, mean_absolute_error


# define pyplot parameters
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'axes.labelsize': 15,
          'axes.titlesize':15,
          'xtick.labelsize':15,
          'ytick.labelsize':15}
pylab.rcParams.update(params)


def baseline_rolling_window(a, window):
    bins = np.linspace(start=np.floor(np.min(a)), stop=np.ceil(np.max(a)), num=1+(np.ceil(np.max(a))-np.floor(np.min(a))).astype(np.uint8))
    hist = np.histogram(a, bins=bins)[0]

    windows_hist = np.lib.stride_tricks.sliding_window_view(hist, window_shape=window)
    sum_windows = np.sum(windows_hist, axis=1)
    bin = np.argmax(sum_windows)

    windows_bins = np.lib.stride_tricks.sliding_window_view(bins, window_shape=window)
    return np.average(windows_bins[bin])


def test(expName, upper, var_n):
    preds = []
    targs = []
    id_list = []
    global line

    #=========  Load settings from Config file ==========
    hyperparameters = loadConfig('./config.txt')
    hyperparameters['name'] = expName
    path_dataset = hyperparameters['path_dataset']
    bs = hyperparameters['batch_size']

    #========== Define parameters here ==========
    device = hyperparameters['device']
    assert os.path.isdir('experiences/' + expName)



    # HyperX
    test_set = CLSA_dataloader_test(path_dataset, var_n, upper, hyperparameters=hyperparameters)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=8)

    #=========  Define model ============
    num_classes = 1
    from torch import nn
    import torch.backends.cudnn as cudnn
    from torchvision import models
    from efficientnet_pytorch import EfficientNet
    from efficientnet_pytorch.utils import MemoryEfficientSwish
    from models.metaEfficient import get_metaEfficient
    from models.attention_model import AttentionNet
    # net = AttentionNet(2)
    # net = nn.Sequential(nn.AdaptiveAvgPool2d((587, 587)), net)
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

    if hyperparameters['device'] == 'cuda':
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    _, _, net = resumeCheckpoint(net, hyperparameters)


    net.eval()
    with torch.no_grad():
        for (inputs, targets, meta, id) in tqdm(test_loader):
            inputs, targets, meta = inputs.to(device), targets.to(device), meta.to(device)
            # inputs = [inputs, meta]
            # outputs = net(meta)
            outputs = net(inputs)
            # outputs, amap = net(inputs)

            preds.extend(outputs.squeeze().detach().cpu().numpy())
            targs.extend(targets.squeeze().detach().cpu().numpy())
            id_list.extend(id)

            # Saliency Batch Size must be 1

            # amap = cv.resize(amap1.cpu().detach().numpy().squeeze(), (587, 587))
            # new_amap = ((amap - amap.min()) * (1/(amap.max() - amap.min()) * 255)).astype('uint8')
            # print(ori.shape)
            # org_im = Image.fromarray(cv.resize(ori.cpu().detach().numpy().squeeze(), (587, 587)))
            # cmaps= ['spring']#'spring', 'seismic',
            # for cmap in cmaps:
            #     print(cmap)
            #     color_map = cm.get_cmap(cmap)
            #     heat = color_map(new_amap)
            #     # Change alpha channel in colormap to make sure original image is displayed
            #     heatmap = copy.copy(heat)
            #     heatmap[:, :, 3] = 0.7
            #     heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
            #
            #     # Apply heatmap on iamge
            #     heatmap_on_image = Image.new("RGBA", org_im.size)
            #     heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
            #     heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
            #     # heat = cv.applyColorMap(new_amap, cv.COLORMAP_PINK)
            #     # fin = cv.addWeighted(heat[:, :, 0:3], 0.5, cv.resize(ori.cpu().detach().numpy().squeeze(), (587, 587)), 0.5, 0)
            #     # heatmap_on_image.show()
            #     # cv.imshow("test", np.array(heatmap_on_image))
            #     cv.imshow("test", np.hstack([np.array(org_im), (heat[:, :, 0:3]*255).astype(np.uint8), np.array(heatmap_on_image)[:, :, 0:3]]))
            #     cv.waitKeyEx()


    import matplotlib.pyplot as plt

    preds = np.array(preds, dtype=np.float32).squeeze()#-100)/10
    targs = np.array(targs, dtype=np.float32).squeeze()#-100)/10
    id = np.array(id_list).squeeze()


    # COMBINE PRED FROM 2 FUNDUS INTO ONE FOR CLASSIFICATION TASK
    # df = pd.DataFrame(id)
    # df[1] = preds[:, 0]
    # df[2] = preds[:, 1]
    # df[3] = df.groupby(0)[1].transform('mean')
    # df[4] = df.groupby(0)[2].transform('mean')
    # df[5] = targs[:, 0]
    # df[6] = targs[:, 1]
    # df = df.drop([1, 2], axis=1)
    # df = df.drop_duplicates()
    # preds = np.array([df[3],df[4]]).transpose()
    # targs = np.array([df[5],df[6]]).transpose()

    # COMBINE PRED FROM 2 FUNDUS INTO ONE FOR REGRESSION TASK
    import pandas as pd
    df = pd.DataFrame(id)
    df[1] = preds
    df[2] = df.groupby(0)[1].transform('mean')
    df[3] = targs
    df = df.drop([1], axis=1)
    df = df.drop_duplicates()
    preds = np.array(df[2])
    targs = np.array(df[3])

    # ROC Curve
    # fpr_real, tpr_real, _ = roc_curve(targs.argmax(axis=1), preds[:,1], drop_intermediate=True)
    # roc = roc_auc_score(np.array(targs).argmax(axis=1), np.array(preds).argmax(axis=1))
    # fpr_50, tpr_50, _ = roc_curve(targs.argmax(axis=1), np.zeros_like(targs.argmax(axis=1)))
    # fig, ax = plt.subplots()
    # ax.plot(fpr_real, tpr_real, 'b-', label=('Sex (AUC=' + str(roc)[0:4]+')'))
    # ax.plot(fpr_50, tpr_50, 'k:')
    # legend = ax.legend(loc='lower right', fontsize='x-large')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.show()
    # np.save("fpr_real_apoe4_combined.npy", fpr_real)
    # np.save("tpr_real_apoe4_combined.npy", tpr_real)


    # Bootstrap for CIs for classification
    # det_roc = []
    # for i in range(2000):
    #     indices = np.array([(random.randrange(0, len(preds))) for i in range(len(preds))])
    #     det_roc.append(roc_auc_score(np.array(targs[indices]).argmax(axis=1), np.array(preds[indices]).argmax(axis=1)))
    # det_roc = np.array(det_roc)
    # alpha = 0.95
    # # R2
    # p = ((1.0-alpha)/2.0) * 100
    # lower_r = max(0.0, np.percentile(det_roc, p))
    # p = (alpha+((1.0-alpha)/2.0)) * 100
    # upper_r = min(1.0, np.percentile(det_roc, p))
    # print('%.1f confidence interval for R2 %.3f%% and %.3f%%' % (alpha*100, lower_r, upper_r))
    #
    # time.sleep(1)
    # print("Model Roc: " + str(det_roc))


    # Bootstrap for CIs for regression
    det_r = []
    det_mae = []
    for i in range(2000):
        indices = np.array([(random.randrange(0, len(preds))) for i in range(len(preds))])
        det_r.append(r2_score(targs[indices],  preds[indices]))
        det_mae.append(mean_absolute_error(targs[indices], preds[indices]))
    det_r = np.array(det_r)
    det_mae = np.array(det_mae)
    alpha = 0.95
    # R2
    p = ((1.0-alpha)/2.0) * 100
    lower_r = max(0.0, np.percentile(det_r, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_r = min(1.0, np.percentile(det_r, p))
    print('%.1f confidence interval for R2 %.3f%% and %.3f%%' % (alpha*100, lower_r, upper_r))
    # MAE
    p = ((1.0-alpha)/2.0) * 100
    lower_m = np.percentile(det_mae, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_m = np.percentile(det_mae, p)
    print('%.1f confidence interval for MAE %.2f%% and %.2f%%' % (alpha*100, lower_m, upper_m))

    mean_targs = np.ones_like(targs) * np.mean(targs)
    mae = mean_absolute_error(targs, preds)
    r2 = r2_score(targs,  preds)
    mae_baseline = mean_absolute_error(targs, mean_targs)
    r2_baseline = r2_score(targs, mean_targs)

    time.sleep(1)
    print("Model R2: " + str(r2))
    print("Model MAE: " + str(mae))

    print("Baseline R2: " + str(r2_baseline))
    print("Baseline MAE: " + str(mae_baseline))
    #
    #
    # total = len(preds)
    # one_counter = 0
    # three_counter = 0
    # five_counter = 0
    # ten_counter = 0
    # fifteen_counter = 0
    # for i, pred in enumerate(preds):
    #     diff = np.abs(pred - targs[i])
    #     if diff <= 1:
    #         one_counter += 1
    #     if diff <= 3:
    #         three_counter += 1
    #     if diff <= 5:
    #         five_counter += 1
    #     if diff <= 10:
    #         ten_counter += 1
    #     if diff <= 15:
    #         fifteen_counter += 1
    # print("Model +/- 1 : " + str(one_counter/total*100))
    # print("Model +/- 3 : " + str(three_counter/total*100))
    # print("Model +/- 5 : " + str(five_counter/total*100))
    # print("Model +/- 10 : " + str(ten_counter/total*100))
    # print("Model +/- 15 : " + str(fifteen_counter/total*100))
    #
    # # calculate baseline accuracy
    # b1 = baseline_rolling_window(targs, 2)
    # b3 = baseline_rolling_window(targs, 6)
    # b5 = baseline_rolling_window(targs, 10)
    # b10 = baseline_rolling_window(targs, 20)
    # b15 = baseline_rolling_window(targs, 30)
    # b_one_counter = 0
    # b_three_counter = 0
    # b_five_counter = 0
    # b_ten_counter = 0
    # b_fifteen_counter = 0
    # for targ in targs:
    #     if np.abs(targ-b1) <= 1:
    #         b_one_counter += 1
    #     if np.abs(targ-b3) <= 3:
    #         b_three_counter += 1
    #     if np.abs(targ-b5) <= 5:
    #         b_five_counter += 1
    #     if np.abs(targ-b10) <= 10:
    #         b_ten_counter += 1
    #     if np.abs(targ-b15) <= 15:
    #         b_fifteen_counter += 1
    # print("Baseline +/- 1 : " + str(b_one_counter/total*100))
    # print("Baseline +/- 3 : " + str(b_three_counter/total*100))
    # print("Baseline +/- 5 : " + str(b_five_counter/total*100))
    # print("Baseline +/- 10 : " + str(b_ten_counter/total*100))
    # print("Baseline +/- 15 : " + str(b_fifteen_counter/total*100))
    #
    # print("P-value +/- 1 : " + str(stats.binom_test(one_counter, total, b_one_counter/total, 'greater')))
    # print("P-value +/- 3 : " + str(stats.binom_test(three_counter, total, b_three_counter/total, 'greater')))
    # print("P-value +/- 5 : " + str(stats.binom_test(five_counter, total, b_five_counter/total, 'greater')))
    # print("P-value +/- 10 : " + str(stats.binom_test(ten_counter, total, b_ten_counter/total, 'greater')))
    # print("P-value +/- 15 : " + str(stats.binom_test(fifteen_counter, total, b_fifteen_counter/total, 'greater')))

    # results = np.array([
    #     lower_r,
    #     upper_r,
    #     lower_m,
    #     upper_m,
    #     r2,
    #     mae,
    #     r2_baseline,
    #     mae_baseline,
    #     one_counter/total*100,
    #     three_counter/total*100,
    #     five_counter/total*100,
    #     ten_counter/total*100,
    #     fifteen_counter/total*100,
    #     b_one_counter/total*100,
    #     b_three_counter/total*100,
    #     b_five_counter/total*100,
    #     b_ten_counter/total*100,
    #     b_fifteen_counter/total*100,
    #     stats.binom_test(one_counter, total, b_one_counter/total, 'greater'),
    #     stats.binom_test(three_counter, total, b_three_counter/total, 'greater'),
    #     stats.binom_test(five_counter, total, b_five_counter/total, 'greater'),
    #     stats.binom_test(ten_counter, total, b_ten_counter/total, 'greater'),
    #     stats.binom_test(fifteen_counter, total, b_fifteen_counter/total, 'greater')
    # ])
    # np.savetxt(expName + "_results.csv", results, delimiter=",")


if __name__ == '__main__':
    var_n = 28
    expName = "Test_EfficientNetB3_0.001_32_CLSA_CFA_GRAHAM_GoodAndUsable"
    upper = 21860
    test(expName, upper, var_n)