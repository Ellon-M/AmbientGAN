## Author: 2022 Ellon


# Lint as python3
import sys
import numpy as np
from tokenizer.tokenize import NoteTokenizer
import time
import datetime
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import *


# helpers
def getNotes():
    """ fetches stored notes
    """

  noteTokenizer = NoteTokenizer()
  notes_arr = noteTokenizer.lspToPy('AmbientGAN/ambience/fore.lsp')
  notes = notes_arr[0]

  return notes

def get_norm_vals():
    """ returns the mean and sd that normalized the input
    """

    notetokenizer = NoteTokenizer()
    notes = getNotes()
    n_vocab = len(set(notes))
    notes_dict = notetokenizer.tokenize(notes)

    X_train, y_train = notetokenizer.prepNoteSequences(notes, notes_dict, 50, n_vocab)

    idx = np.random.randint(0, X_train.shape[0], 32)
    real_sequences = X_train[idx]
    real_sequences = torch.Tensor(real_sequences).cuda()
    mu = torch.mean(real_sequences)
    sd = torch.std(real_sequences)
    norm_vals ={}
    norm_vals['mean'] = mu
    norm_vals['sd'] = sd

    return norm_vals



# network

class Self_Attn(nn.Module):
  """ Self Attention Layer """

  def __init__(self, in_dim, k):
    super(Self_Attn, self).__init__()
    self.channel_in = in_dim
    self.k = k

    self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//k, kernel_size=1)
    self.key_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim//k , kernel_size=1)
    self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//k, kernel_size=1)
    self.self_att = nn.Conv2d(in_dim//k, in_dim, 1)

    self.gamma = nn.Parameter(torch.zeros(1))
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    """
       inputs :
                x : input feature map [Batch, Channel, Height, Width]
       returns :
                out : self attention value + input feature
                o: [Batch, Channel, Height, Width]
    """

    m_batchsize, C, height, width = x.size()
    proj_query = self.query_conv(x).view(m_batchsize, -1, height*width) # B X CX(N)
    proj_key = self.key_conv(x).view(m_batchsize, -1, height*width) # B X C x (*W*H)
    proj_value = self.value_conv(x).view(m_batchsize, -1, height*width)
    energy =  torch.bmm(proj_key.permute(0,2,1), proj_query) # scores
    beta = self.softmax(energy)
    v = torch.bmm(proj_value, beta)
    value = v.view(m_batchsize, -1, width, height)

    o = self.self_att(value)

    out = self.gamma * o + x
    return out, o



class Generator(nn.Module):
    """ Generator model """

    def __init__(self, out_dim, rows):
      super(Generator, self).__init__()
      self.out = out_dim
      self.seq_length = rows
      self.seq_shape =(self.seq_length, 1)
      layer1 = []
      layer2 = []
      layer3 = []
      last = []

      layer1.append(nn.Linear(1000, out_dim))
      layer1.append(nn.BatchNorm1d(out_dim, momentum=0.8))
      layer1.append(nn.LeakyReLU(negative_slope=0.1))

      out_dim = out_dim * 2

      layer2.append(nn.Linear(int(out_dim/2), out_dim))
      layer2.append(nn.BatchNorm1d(out_dim, momentum=0.8))
      layer2.append(nn.LeakyReLU(negative_slope=0.1))


      layer3.append(nn.Linear(out_dim, out_dim))
      layer3.append(nn.BatchNorm1d(out_dim, momentum=0.8))
      layer3.append(nn.LeakyReLU(negative_slope=0.1))

      out_dim = out_dim * 2

      self.l1 = nn.Sequential(*layer1)
      self.l2 = nn.Sequential(*layer2)
      self.l3 = nn.Sequential(*layer3)

      last.append(nn.Linear(int(out_dim/2), np.prod(self.seq_shape)))
      last.append(nn.Tanh())

      self.last = nn.Sequential(*last)

#       self.attn = Self_Attn(512, 8)


    def forward(self, z):
      z = z.view(128, 1000)
      out = self.l1(z)
      out = out.view(128, int(self.out))
      out = self.l2(out)
      out = out.view(128, int(self.out * 2))
      out = self.l3(out)
#       out = out.view(32, -1, 1, 1)
#       out, p1 = self.attn(out)
#       out = torch.flatten(out, 1)
      out = self.last(out)
#       return out.squeeze(), p1
      return out.squeeze()



class Discriminator(nn.Module):
  """ Discriminator Model """

  def __init__(self, out_dim, rows):
    super(Discriminator, self).__init__()
    self.seq_length = rows
    self.out = out_dim
    self.seq_shape =(self.seq_length, )

    self.l1 = nn.ModuleDict({
        'uni-lstm': nn.LSTM(
            input_size=1,
            hidden_size=out_dim,
            batch_first = True
        ),
        'bi-lstm': nn.LSTM(
            input_size=out_dim,
            hidden_size= int(out_dim/2),
            batch_first = True,
            bidirectional=True
        )
    })

    layer2 = []
    layer3 = []
    last = []

    layer2.append(nn.Linear(out_dim, out_dim))
    layer2.append(nn.LeakyReLU(negative_slope=0.1))

    out_dim = out_dim / 2

    layer3.append(nn.Linear(int(out_dim) * 2, int(out_dim)))
    layer3.append(nn.LeakyReLU(negative_slope=0.1))

#     last.append(nn.Linear(int(out_dim) * int(self.seq_length) *  int(self.seq_length), 1))
    last.append(nn.Linear(int(out_dim), 1))
    last.append(nn.Sigmoid())

    self.l2 = nn.Sequential(*layer2)
    self.l3 = nn.Sequential(*layer3)
    self.last = nn.Sequential(*last)

#     self.attn = Self_Attn(256, 8)


  def forward(self, x):
    x = x.view(128, self.seq_length, 1)
    out = self.l1['uni-lstm'](x)[0]
    out = out.view(128, self.seq_length, self.out)
    out = self.l1['bi-lstm'](out)[0]
    out = out.view(128, self.seq_length, int(self.out))
    out = self.l2(out)
    out = self.l3(out)
#     out = out.view(32, int(self.out / 2), self.seq_length,  1)
#     out, p1 = self.attn(out)
#     out = torch.flatten(out, 1)
#     out = out.view(32, self.seq_length, int(self.out / 2), self.seq_length)
    out = self.last(out)
#     print(out.squeeze())

#     return out.squeeze(), p1
    return out.squeeze()




class SAGAN(object):
  """ trainer class """

  def __init__(self, config):
      self.model_save_step = config.save_step
      self.model_save_path = config.save_path
      self.pretrained_model = config.pretrained
      self.lambda_gp = config.lambda_gp
      self.d_int = 1
      self.d_loss = []
      self.g_loss = []
      self.gen_train_step = config.gen_train_step
      self.epochs = config.epochs
      self.interval = config.progress
      self.batch_size = config.batch_size

      torch.cuda.empty_cache()  # clear GPU cache

      self.build()

      if self.pretrained_model:
        self.load_pretrained_model()


  def build(self):

      self.G = Generator(256, 1000).cuda()
      self.D = Discriminator(512, 1000).cuda()

      self.g_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
      self.d_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

      self.criterion = nn.BCELoss()
      self.c_loss = torch.nn.CrossEntropyLoss()

      print(self.G)
      print(self.D)

  def resetGrad(self):
      self.d_opt.zero_grad()
      self.g_opt.zero_grad()
      self.G.zero_grad()
      self.D.zero_grad()

  def train(self):
      """ trains both the Discriminator and the Generator
       """
      #load, convert the data
      notetokenizer = NoteTokenizer()
      notes = getNotes()
      n_vocab = len(set(notes))
      notes_dict = notetokenizer.tokenize(notes)

      X_train, y_train = notetokenizer.prepNoteSequences(notes, notes_dict, 1000, n_vocab)


      # Start with trained model
      if self.pretrained_model:
          start = self.pretrained_model + 1
      else:
          start = 0

      #start train
      start_time = time.time()
      for epoch in range(start, self.epochs):

          print("Epoch ", epoch + 1)

          # real random sequences
          idx = np.random.randint(0, X_train.shape[0], batch_size)
          real_sequences = X_train[idx]

          # adv ground truths
          real = torch.ones(self.batch_size, 1000).cuda()
          fake = torch.zeros(self.batch_size, 1000).cuda()

          #----training D----

          self.D.train()

          real_sequences = torch.Tensor(real_sequences).cuda()
          mu = torch.mean(real_sequences)
          sd = torch.std(real_sequences)
          normalized_res = (real_sequences - mu)/sd


          # train the discriminator on real sequences
          # D_out_real, dr = self.D(real_sequences)   # attn
          D_out_real = self.D(normalized_res)

          # hinge
          D_loss_real = torch.nn.ReLU()(1.0 - D_out_real).mean()
          D_out_loss = self.criterion(D_out_real, real)
          # print("D loss real", D_loss_real.data)


          z = torch.randn(batch_size, 1000).cuda()
          # fake_sequences, gf = self.G(z)    # attn
          fake_sequences = self.G(z)
          # D_out_fake, df = self.D(fake_sequences)   # attn
          D_out_fake = self.D(fake_sequences)


          # hinge
          # D_loss_fake = D_out_fake.mean()
          D_loss_fake = torch.nn.ReLU()(1.0 + D_out_fake).mean()
          D_z_loss = self.criterion(D_out_fake, fake)


          D_loss = D_out_loss + D_z_loss # network weights updated
          self.resetGrad()
          D_loss.backward()
          self.d_opt.step()


          #----training G----
          if epoch % self.d_int == 0:
              # random noise
              noise = torch.randn(batch_size, 1000).cuda()
              # fake_sequences, _ = self.G(noise)   # attn
              fake_sequences = self.G(noise)


              # loss with fake sequences
              G_out_fake = self.D(fake_sequences)
              # G_out_fake_avg = torch.div(sum((G_out_fake)), len((G_out_fake)))
#               G_loss = - G_out_fake.mean()
              G_loss = self.criterion(G_out_fake, real)
    

              self.resetGrad()
              G_loss.backward()
              self.g_opt.step()

          # train the generator after k steps
          if epoch % self.gen_train_step == 0:
            self.G.train()

          # progress
          if (epoch + 1) % self.interval == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], D Loss: {:.4f}, G Loss: {:.4f}"
                  .format(elapsed, epoch + 1, epochs, (epoch + 1),
                               epochs, D_out_loss, D_z_loss
                              )) #self.G.attn.gamma.mean().data

          # save models
          if (epoch + 1) % self.model_save_step == 0:
            torch.save(self.G.state_dict(),
                       os.path.join(self.model_save_path, '{}_G.pth'.format(epoch + 1)))
            torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(epoch + 1)))

          # append losses to loss lists
          if (epoch + 1) % self.d_int == 0:
              self.d_loss.append(D_loss_real.cpu().detach().numpy())
              self.g_loss.append(G_loss.cpu().detach().numpy())



      self.plot_loss()


  def load_pretrained_model(self):
    self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
    self.D.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
    print('loaded trained models (Epoch: {})..!'.format(self.pretrained_model))


  def plot_loss(self):
    plt.plot(self.d_loss, c='red')
    plt.plot(self.g_loss, c='blue')
    plt.title('Loss per Epoch')
    plt.legend(['Discriminator', 'Generator'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_per_epoch.png', transparent=True)
    plt.show()
    plt.close()


