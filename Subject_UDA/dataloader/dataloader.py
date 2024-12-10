from torch.utils.data import Dataset
import numpy as np
import torch


class BuildDataset(Dataset):
    def __init__(self, data_path, dataset):
        self.data_path = data_path
        self.dataset = dataset
        self.len = len(self.data_path[0])

    def __getitem__(self, index):
        x_data = np.load(self.data_path[0][index])
        y_data = np.load(self.data_path[1][index])
        x_data = torch.from_numpy(np.array(x_data).astype(np.float32))
        y_data = torch.from_numpy(np.array(y_data).astype(np.float32))
        if self.dataset == "ISRUC":
            eog = x_data[:, :2, :]
            eeg = x_data[:, 2:, :]
        elif self.dataset == "Hang7":
            eog = x_data[:, 6:, :]
            eeg = x_data[:, :6, :]
        elif self.dataset == "HMC":
            eog = x_data[:, 4:, :]
            eeg = x_data[:, :4, :]
        elif self.dataset == "SleepEDF":
            eeg = x_data[:, 0:2, :]
            eog = x_data[:, 2, :].view(20, 1, -1)
        else:
            eog = torch.concat((x_data[:, 0, :].view(20, 1, -1), x_data[:, 2:4, :],
                                x_data[:, 9, :].view(20, 1, -1)), dim=1)
            eeg = torch.concat((x_data[:, 1, :].view(20, 1, -1), x_data[:, 4:9, :]), dim=1)
        return eog, eeg, y_data

    def __len__(self):
        return self.len


class Builder(object):
    def __init__(self, data_path, dataset):
        super(Builder, self).__init__()
        self.data_set = dataset
        self.data_path = data_path
        self.Dataset = BuildDataset(self.data_path, self.data_set)
