import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

def get_metaEfficient(n_features):
    return Net(arch=EfficientNet.from_pretrained('efficientnet-b3'),n_meta_features= n_features)

class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = nn.Sequential(nn.AdaptiveAvgPool2d((300, 300)), arch, MemoryEfficientSwish())
        self.arch._fc = nn.Sequential(nn.Linear(in_features=1000, out_features=500, bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.2))
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.output = nn.Sequential(
            nn.Linear(500 + 250, 1),
            nn.ReLU()
        )

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta.float())
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.output(features)
        return output

