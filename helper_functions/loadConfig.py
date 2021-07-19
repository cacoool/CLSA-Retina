from six.moves import configparser
import torch

def loadConfig(path):
    #=========  Load settings from Config file
    config = configparser.RawConfigParser()
    config.read(path)

    #[data paths]
    path_dataset = config.get('data paths', 'path_dataset')

    #[experiment name]
    name = config.get('experiment name', 'name')


    #[training settings]
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    lr = float(config.get('training settings', 'lr'))
    flip_augmentation = config.getboolean('training settings', 'flip_augmentation')
    affine_augmentation = config.getboolean('training settings', 'affine_augmentation')
    mixture_augmentation = config.getboolean('training settings', 'mixture_augmentation')


    # hyperparameters
    hyperparameters = {
        "path_dataset":  path_dataset,

        "name": name,

        "N_epochs": N_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "flip_augmentation": flip_augmentation,
        "affine_augmentation": affine_augmentation,
        "mixture_augmentation": mixture_augmentation,

        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return hyperparameters


if __name__ == '__main__':
    loadConfig('../config.txt')