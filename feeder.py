from os import WIFCONTINUED
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Feeder(Dataset):
    """
    load processed skeleton dataset
    """
    def __init__(self, data_path, label_path, split_csv,
                 frame_annota_path, window_size = 6000, use_mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.use_mmap = use_mmap
        self.split_csv = split_csv
        self.frame_annota_path = frame_annota_path
        self.window_size = window_size
        self._load_data()

    def _load_data(self):
        """
        data shape: N C V T
        """
        # load label
        csv_info = pd.read_csv(self.split_csv, encoding='utf-8')
        self.frame_annotations = pd.read_csv(self.frame_annota_path, encoding='gbk')

        try:
            with open(self.label_path) as f:
                sample_name, label = pickle.load(f)
        except:
            with open(self.label_path, 'rb') as f:
                sample_name, label = pickle.load(f, encoding='latin1')
        # load data
        if self.use_mmap:
            data = np.load(self.data_path, mmap_mode='r')
        else:
            data = np.load(self.data_path)
        self.data = np.zeros((len(csv_info.index), data.shape[1], data.shape[2], data.shape[3]))
        self.sample_name = []
        self.label = []
        for i, idx in enumerate(csv_info["index"]):
            idx = int(idx)
            self.data[i, :, :, :] = data[idx, :, :, :]
            self.sample_name.append(sample_name[idx])
            self.label.append(self._label_map(label[idx]))

    def _label_map(self, label):
        if label=='F-':
            return 0
        else:
            return 1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if self.window_size > 0:
            data_numpy = self.auto_pading(data_numpy, self.window_size)

        return data_numpy, label, index

    def auto_pading(self, data_numpy, size, random_pad=False):
        C, T, V = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((C, size, V))
            data_numpy_paded[:, begin:begin + T, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy[:, :size, :]
