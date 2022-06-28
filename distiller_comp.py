import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math
from einops import rearrange
from torch import einsum
from torch.nn.modules import loss
import logging
'''Set up a module logger'''
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(dim_out)#Normalize(2)        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x)
        return x

class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()
        self.n     = args.n
        self.m     = args.m
        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()
        self.comp = args.comp

        self.t_net = t_net
        self.s_net = s_net


    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        feat_num = len(t_feats)
        s_feats, s_out = self.s_net.extract_feature(x)

        loss_distill = 0

        loss_distill =self.distillation_loss_all(s_feats[-1],t_feats[-1])

        if type(loss_distill) == list:
            loss_distill = sum([i for i in loss_distill]) 
        return s_out, loss_distill


    def distillation_loss_all(self,f_s,f_t):

        if len(f_s.size())==3:
            f_s = f_s.unsqueeze(0)
        if len(f_t.size())==3:
            f_t = f_t.unsqueeze(0)

        if self.comp == 'kd':
            return 0.01*self.kd_loss(f_s,f_t)
        elif self.comp == 'at':
            return self.at_loss(f_s,f_t)
        elif self.comp =='fitnet':
            return self.hintloss(f_s,f_t)
        elif self.comp =='ch':
            return 0.5*self.ch_loss(f_s,f_t)

    def kd_loss(self,f_s,f_t):
        T = 7
        p_s = F.log_softmax(f_s/T, dim=-1)
        p_t = F.softmax(f_t/T, dim=-1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / f_s.shape[0]
        return loss

    def hintloss(self,f_s,f_t):
        return nn.MSELoss()(f_s,f_t)

    def at_loss(self, f_s,f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        return (self.at(f_s) - self.at(f_t)).pow(2)

    def at(self, f):
        p=2
        return F.normalize(f.pow(p).mean(1).view(f.size(0), -1))

    def ch_loss(self,f_s,f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = self.grid_tensor_seq(f_s,self.n,self.m)
        f_t = self.grid_tensor_seq(f_t,self.n,self.m)

        s_ch = einsum('b i d, b j d ->b i j',f_s,f_s)
        s_ch = torch.nn.functional.normalize(s_ch, dim = 2)

        t_ch = einsum('b i d, b j d ->b i j',f_t,f_t)
        t_ch = torch.nn.functional.normalize(t_ch, dim = 2)

        G_diff = s_ch - t_ch
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss

    def grid_tensor_seq(self,x,n,m):
        b,c,h,w = x.shape
        k_size_h = h//n
        stride_h = k_size_h

        k_size_w = w//m
        stride_w = k_size_w

        temp = x.unfold(2,k_size_h,stride_h).unfold(3,k_size_w,stride_w)

        temp = temp.contiguous().view(temp.size(0),temp.size(1),-1,temp.size(4),temp.size(5))
        temp = temp.permute(0,2,1,3,4)
        temp = temp.contiguous().view(-1,temp.size(2),temp.size(3),temp.size(4))
        return temp.view(temp.size(0),temp.size(1),-1)

