import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

seed = 2020
np.random.seed(seed)


def get_data_loader(is_train, batch_size, image_transform, dataset='cedar'):
    if dataset == 'cedar':
        data_dir = './data/cedar'
    elif dataset == 'bengali':
        data_dir = './data/BHSig260/Bengali'
    elif dataset == 'hindi':
        data_dir = './data/BHSig260/Hindi'
    else:
        raise ValueError(f'Unknow dataset {dataset}')
    data = SignDataset(is_train, data_dir, image_transform)
    is_shuffle = is_train
    loader = DataLoader(data, batch_size=batch_size, shuffle=is_shuffle, num_workers=10, pin_memory=True)
    return loader


class SignDataset(Dataset):
    def __init__(self, is_train: bool, data_dir: str, image_transform=None):
        if not os.path.exists(os.path.join(data_dir, 'train.csv')) or not os.path.exists(
                os.path.join(data_dir, 'test.csv')):
            print('Not found train/test splits, run create_annotation first')
        else:
            print('Use existed train/test splits')

        if is_train:
            self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)
        else:
            self.df = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2, y = self.df.iloc[index]

        x1 = Image.open(x1).convert('L')
        x2 = Image.open(x2).convert('L')

        if self.image_transform:
            x1 = self.image_transform(x1)
            x2 = self.image_transform(x2)

        return x1, x2, y