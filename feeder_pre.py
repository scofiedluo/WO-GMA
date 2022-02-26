from os import WIFCONTINUED
import torch
import pickle
import numpy as np
import random
from torch.utils import data
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, window_size = 1000, use_mmap=True):
        """

        """
        self.data_path = data_path
        self.label_path = label_path
        self.use_mmap = use_mmap
        self.window_size = window_size
        self.load_data()

    def load_data(self):
        # data shape: N C V T

        # load label
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if self.window_size > 0:
            data_numpy = self.auto_pading(data_numpy,self.window_size)

        return data_numpy, label, index
    

    def auto_pading(self, data_numpy, size, random_pad=False):
        C, T, V = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((C, size, V))
            data_numpy_paded[:, begin:begin + T, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy