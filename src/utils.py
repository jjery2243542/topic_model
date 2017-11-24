import numpy as np
from torch.utils import data
import torch 
from collections import namedtuple
import json
from tensorboardX import SummaryWriter
class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'batch_size',
            'epochs',
            ]
        )
        default = [1e-2, 32, 300]
        self._hps = self.hps._make(default)

    def get_tuple(self):
        return self._hps

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_dict = json.load(f_json)
        self._hps = self.hps(**hps_dict)

    def dump(self, path):
        with open(path, 'w') as f_json:
            json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))

class DataLoader(data.Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

def get_loader(npy_path, batch_size=32, num_workers=2):
    dataset = DataLoader(npy_path)
    data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

    return data_loader


class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

if __name__ == '__main__':
    hps = Hps()
    hps.dump('./hps/v1.json')
    #data_loader = get_loader('../dataset/20news/npy/train.npy')
    #for batch in data_loader:
    #    print(torch.sum(batch, dim=1))
