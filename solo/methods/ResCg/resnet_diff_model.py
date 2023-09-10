"Only channel Gating Krishna et. al (Dynamic Channle selection in self-supervised learning)"


import math
import logging
import torch
import torch.nn as nn
from prettytable import PrettyTable
from collections import deque
import sys
# sys.path.insert(0, "/home/tarun/Documents/PhD/DGC_SSL")
torch.cuda.manual_seed(42)
from solo.methods.ResCg.mask import Mask_c

import solo.methods.ResCg.resnet_gating as resnet18_cg
import solo.methods.ResCg.resnet_vicreg as resnet18_vanilla

class ResDG(nn.Module):

    # def __init__(self, block, layers, h=224, w=224, num_classes=1000,
    #              zero_init_residual=False, groups=1, width_per_group=64,
    #              replace_stride_with_dilation=None, norm_layer=None, **kwargs):
    
    def __init__(self, h, w, **kwargs):

        super(ResDG, self).__init__()
        print(h,w)
        print("First====>", kwargs.keys())
        self.model_1 = resnet18_cg.resdg18(h=h, w=w, **kwargs)
        print("Second===>", kwargs.keys())
        self.model_2 = resnet18_vanilla.resdg18(h=h, w=w, **kwargs)

    def forward(self, x, flag, den_target, lbda, gamma, p):

        if flag==0:

            out = self.model_1(x, flag, den_target, lbda, gamma, p)
            return out
        else:
            out = self.model_2(x)
            return out 


        

class ExpAnnealing(object):
    r"""
    Args:
        T_max (int): Maximum number of iterations.
        eta_ini (float): Initial density. Default: 1.
        eta_min (float): Minimum density. Default: 0.
    """

    def __init__(self, T_ini, eta_ini=1, eta_final=0, up=False, alpha=1):
        self.T_ini = T_ini
        self.eta_final = eta_final
        self.eta_ini = eta_ini
        self.up = up
        self.last_epoch = 0
        self.alpha = alpha

    def get_lr(self, epoch):
        if epoch < self.T_ini:
            return self.eta_ini
        elif self.up:
            return self.eta_ini + (self.eta_final-self.eta_ini) * (1-
                   math.exp(-self.alpha*(epoch-self.T_ini)))
        else:
            return self.eta_final + (self.eta_ini-self.eta_final) * math.exp(
                   -self.alpha*(epoch-self.T_ini))

    def step(self):
        self.last_epoch += 1
        return self.get_lr(self.last_epoch)

if __name__ == "__main__":

    from solo.losses.regularization_channel import*  

    model = resdg18(**{"cifar": True})
    import torchsummary as summary
    
    criterion = Loss()
    model.set_criterion(criterion)
    y = torch.rand(8, 3, 32, 32)
    target = torch.ones(8)
    den_target = 0.5
    lbda = 5
    gamma =1
    alpha=0.02
    p_anneal = ExpAnnealing(0, 1, 0, alpha=alpha)
    p = p_anneal.get_lr(2)
    # param_dict = dict(model.named_parameters())
    # # print(param_dict.keys())
    # params = []
    # BN_name_pool = []
    # for m_name, m in model.named_modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print(m_name, m)
    #         BN_name_pool.append(m_name + '.weight')
    #         BN_name_pool.append(m_name + '.bias')

    model.train()
    # summary(model, (3, 32, 32))
    output = model(y,  den_target, lbda, gamma, p)
    
    batch_size =8
    flops_real = output["flops_real"]  
    flops_mask = output["flops_mask"]  
    flops_ori =  output["flops_ori"] 


    flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        # block flops
    flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
    flops_mask = flops_mask.mean(0).sum()
    flops_ori = flops_ori.mean(0).sum() + flops_conv1.mean() + flops_fc.mean()
    flops_real = flops_conv + flops_mask + flops_conv1.mean() + flops_fc.mean()
    print(flops_real, flops_ori, flops_real/flops_ori)
    # print(output)

