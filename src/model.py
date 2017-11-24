import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class D_VAE(nn.Module):
    def __init__(self, h_dim=512, vocab_size=2000, n_z=64, z_dim=512):
        super(D_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, n_z))
        
        self.topic_emb = nn.Linear(n_z, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, vocab_size),
            nn.Sigmoid())
   
    def forward(self, x, tau=1.):
        z_logits = self.encoder(x)
        z_distr = F.softmax(z_logits)
        z_soft_distr = F.softmax(z_logits / tau)
        topic_vec = self.topic_emb(z_soft_distr)
        x_prime = self.decoder(topic_vec)
        return x_prime, z_distr

if __name__ == '__main__':
    model = D_vae()
    x = Variable(torch.randn(16, 5000))
    x_prime, z_distr = model(x)
