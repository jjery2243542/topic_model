from model import D_VAE
from utils import get_loader
from utils import Hps
from utils import Logger
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.ae = None
        self._build_model()
        self.logger = Logger(log_dir)

    def _build_model(self):
        self.ae = D_VAE()
        if torch.cuda.is_available():
            self.ae.cuda()
        self.opt = optim.Adam(self.ae.parameters(), lr=self.hps.lr)

    def train(self, model_path):
        epochs = self.hps.epochs
        iterations = len(self.data_loader)
        for epoch in range(epochs):
            for i, x in enumerate(self.data_loader):
                x = Variable(x, requires_grad=True)
                x_prime, z_distr = self.ae(x)
                loss = nn.CrossEntropyLoss()
                loss_rec = loss(x_prime, x)
                loss_rec.backward()
                entropy = -torch.mean(z_distr * np.log(z_distr))
                self.opt.step()
                info = {
                    'loss_rec': loss_rec.data[0],
                    'entropy': entropy.data[0],
                }
                slot_value = (epoch, i+1, iterations) + tuple(info.values)
                print('Epoch:%d, Iter:[%05d/%05d], loss_rec:%.3f, entropy:%.3f' % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, i + 1)

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v1.json')
    hps_tuple = hps.get_tuple()
    data_loader = get_loader('../dataset/20news/npy/train.npy') 
    solver = Solver(hps_tuple, data_loader)
