from torch.utils.data import Dataset
import torch
import numpy as np
import zarr
from glob import glob
from PIL import Image
import random
from torchvision import transforms
import h5py

input_size = 300

class CLSA_dataloader_train(Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, dataset, var_n, min, hyperparameters):
        super(CLSA_dataloader_train, self).__init__()
        with h5py.File(dataset, "r") as f:  # "with" close the file after its nested commands
            self.label = f["gt"][0:min][:, var_n]
            self.meta = f["meta"][0:min][:]
            self.data = f["img"][0:min][:]
        print("Training set min : " + str(self.label.min()))
        print("Training set max : " + str(self.label.max()))
        print("Training set avg : " + str(self.label.mean()))
        print("Training set std : " + str(self.label.std()))

        self.len = len(self.label)
        self.hyperparameters = hyperparameters
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = idx
        data = self.data[n][:, :, :]

        data = np.flip(data, axis=2)
        data = Image.fromarray(data)

        # label for regression
        label = torch.tensor(self.label[n])

        # label for apoe4
        # label = torch.tensor(self.label[n]).long()
        # if label == 1:
        #     label = torch.tensor((1, 0))
        # else:
        #     label = torch.tensor((0, 1))

        data = self.trans(data)
        meta = torch.tensor(self.meta[n])

        return data, label, meta.float()

class CLSA_dataloader_valid(Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, dataset, var_n, min, max, hyperparameters):
        super(CLSA_dataloader_valid, self).__init__()
        with h5py.File(dataset, "r") as f:  # "with" close the file after its nested commands
            self.label = f["gt"][min:max][:, var_n]
            self.meta = f["meta"][min:max][:]
            self.data = f["img"][min:max][:]
        print("Validation set min : " + str(self.label.min()))
        print("Validation set max : " + str(self.label.max()))
        print("Validation set avg : " + str(self.label.mean()))
        print("Validation set std : " + str(self.label.std()))

        self.len = len(self.data)
        self.indices = np.array([(random.randrange(0, len(self.data))) for i in range(self.len)])
        self.hyperparameters = hyperparameters
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = idx
        data = self.data[n][:, :, :]
        data = np.flip(data, axis=2)
        data = Image.fromarray(data)

        # label for regression
        label = torch.tensor(self.label[n])

        # label for apoe4
        # label = torch.tensor(self.label[n]).long()
        # if label == 1:
        #     label = torch.tensor((1, 0))
        # else:
        #     label = torch.tensor((0, 1))

        data = self.trans(data)
        meta = torch.tensor(self.meta[n])

        return data, label, meta.float()


class CLSA_dataloader_test(Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, dataset, var_n, max, hyperparameters):
        super(CLSA_dataloader_test, self).__init__()
        with h5py.File(dataset, "r") as f:  # "with" close the file after its nested commands
            self.label = f["gt"][max:][:, var_n]
            self.id = f["gt"][max:][:, 0]
            self.meta = f["meta"][max:][:]
            self.data = f["img"][max:][:]
        print("Testing set min : " + str(self.label.min()))
        print("Testing set max : " + str(self.label.max()))
        print("Testing set avg : " + str(self.label.mean()))
        print("Testing set std : " + str(self.label.std()))

        self.len = len(self.data)
        self.patch_height = 1288
        self.patch_width = 1288
        self.hyperparemeters = hyperparameters
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = idx
        data = self.data[n][:, :, :]
        data = np.flip(data, axis=2)

        data = Image.fromarray(data)

        # label for regression
        label = torch.tensor(self.label[n])

        # label for apoe4
        # label = torch.tensor(self.label[n]).long()
        # if label == 1:
        #     label = torch.tensor((1, 0))
        # else:
        #     label = torch.tensor((0, 1))

        data = self.trans(data)
        meta = torch.tensor(self.meta[n])

        id = self.id[n]

        return data, label, meta, id


if __name__ == '__main__':
    from helper_functions.loadConfig import loadConfig
    hyperparameters = loadConfig('../config.txt')
    path_dataset =  hyperparameters['path_dataset']
    var_n = 1
    lower = 17200
    upper = 20800
    test = CLSA_dataloader_train(path_dataset, var_n, lower, hyperparameters)
