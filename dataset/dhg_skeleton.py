import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from torch.utils.data import DataLoader
from dataset.skeleton import Skeleton


edge = ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21))


class DHG_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False):
        super().__init__(data_path, label_path, window_size, final_size, mode, decouple_spatial, num_skip_frame,
                         random_choose, center_choose)
        self.edge = edge


if __name__ == '__main__':
    data_path = "/home/xzzit/DSTA-Net/SHREC_data/train_skeleton.pkl"
    label_path = "/home/xzzit/DSTA-Net/SHREC_data/train_label_14.pkl"

    loader = DataLoader(
        dataset=DHG_SKE(data_path, label_path, window_size=150, final_size=10, mode='train',
                        random_choose=True, center_choose=False, decouple_spatial=False, num_skip_frame=None),
        batch_size=1,
        shuffle=False,
        num_workers=0)

    for i, (data, label) in enumerate(loader):
        print(data.shape)
        print(label.shape)
        break