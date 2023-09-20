import numpy as np
import pickle
from torch.utils.data import Dataset
from dataset.video_data import *


class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,
                 mode='train', decouple_spatial=False, num_skip_frame=None,
                 random_choose=False, center_choose=False):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.num_skip_frame = num_skip_frame
        self.decouple_spatial = decouple_spatial
        self.edge = None
        self.load_data()

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = int(self.label[index])
        data_numpy = np.array(data_numpy)  # nctv

        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM

        # data transform
        if self.decouple_spatial:
            data_numpy = decouple_spatial(data_numpy, edges=self.edge)
        if self.num_skip_frame is not None:
            velocity = decouple_temporal(data_numpy, self.num_skip_frame)
            C, T, V, M = velocity.shape
            data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)

        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        if self.center_choose:
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)

        if self.mode == 'train':
            return data_numpy.astype(np.float32), label
        else:
            return data_numpy.astype(np.float32), label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)