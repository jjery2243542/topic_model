import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2)
    return Variable(torch.randn(*size)*xavier_stddev, requires_grad=True)

class D_VAE(nn.Module):
    def __init__(self, h_dim=512, vocab_size=5000, n_z=64, z_dim=512):
        super(D_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, n_z))
        
        self.topic_emb = xavier_init(size=[n_z, z_dim])

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, vocab_size),
            nn.Sigmoid())
   
    def forward(self, x, tau):
        z_logits = self.encoder(x)
        z_distr = F.softmax(z_logits)
        z_soft_distr = F.softmax(z_logits / tau)
        topic_vec = torch.mm(z_soft_distr, self.topic_emb)
        x_prime = self.decoder(topic_vec)
        return x_prime, z_distr

if __name__ == '__main__':
    model = D_VAE()
    x = Variable(torch.randn(16, 5000))
    x_prime, z_distr = model(x, 1.0)
    print(x_prime.size(), z_distr.size())
