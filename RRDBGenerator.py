# This file contains the feedback loop module
import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
import RRDB_block as B

import RRDB_options as option

logger = logging.getLogger('base')
####################
# initialize
####################

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm3d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    if init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################
# Generator
# Default
def define_G():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_48():
    return define_G()

def define_G_32():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=32, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_16():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=16, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_64():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=64, # 64 - intermediate reisudal feature of RRDB block
                   nb=6, # 23 - number of RRDB block
                   gc=16, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

# Signature of RRDB
#def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
#            norm_type=None, act_type='leakyrelu', mode='CNA'):
#        super(RRDB, self).__init__()
#        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
#            norm_type, act_type, mode)
#        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
#            norm_type, act_type, mode)
#        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
#            norm_type, act_type, mode)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(RRDBNet, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


def define_SER():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = SERNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_SER_b1b2():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = SERNet_b1b2(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_SER_b1():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = SERNet_b1(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_SER_b2():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = SERNet_b2(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_SER_G():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = SERNet_G(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type='batch', # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG


# This model has se on all 3 positions of that can have se
class SERNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(SERNet, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        se = B.SELayer(nf, reduction=8)
        
        rb_blocks = [B.RRDB_SE(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv, se)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

# This model has se at block level but not global level
class SERNet_b1b2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(SERNet_b1b2, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        rb_blocks = [B.RRDB_SE(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
    
# This model has se at block 1 level but not global level or block 2 level
class SERNet_b1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(SERNet_b1, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        rb_blocks = [B.RRDB_SE_b1(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

# This model has se at block 2 level but not global level or block 1 level
class SERNet_b2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(SERNet_b2, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        rb_blocks = [B.RRDB_SE_b2(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

# This model has se on global but no block level
class SERNet_G(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, norm_type=None, \
            act_type='leakyrelu', mode='CNA'):
        super(SERNet_G, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # Each RRDB block will sum, and each RRDB subblock ResidualDenseBlock_5C will also sum
        # Each RRDB block contain 3 subblock, each subblock contains 5 densely connected convolutions
        
        # In case model is too large, can either 
        # Reduce the number of block - 23 is a lot! Wow. 
        # Reduce the number of subblock in each block, 
        # Reduce the convolution in each sub block
        # Reduce the dense connection. 32 is pretty huge actually for 3D
        
        se = B.SELayer(nf, reduction=8)
        
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv, se)),\
                                  HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

###############################
# Generator without the bn
###############################
def define_G_nobn():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=48, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type=None, # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_32_nobn():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=32, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type=None, # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_16_nobn():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=16, # 64 - intermediate reisudal feature of RRDB block
                   nb=4, # 23 - number of RRDB block
                   gc=12, # 32 - growth rate of dense block
                   norm_type=None, # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG

def define_G_64_nobn():
    opt = option.parse("train_ESRGAN.json", is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt_net = opt['network_G']
    netG = RRDBNet(in_nc=1, # 1
                   out_nc=1, # 1
                   nf=64, # 64 - intermediate reisudal feature of RRDB block
                   nb=6, # 23 - number of RRDB block
                   gc=16, # 32 - growth rate of dense block
                   norm_type=None, # null
                   act_type='leakyrelu', 
                   mode=opt_net['mode']) # CNA
    return netG