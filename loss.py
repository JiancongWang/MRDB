#%% This file implements the loss part of the pytorch version of mDCSRN
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from HighRes3DNet import HighRes3DNet

from discriminator_torch import SSR2Discriminator

from torch.nn.parallel.scatter_gather import scatter, gather

# This loss has 2 parts. The L1/L2 loss and highres3dnet loss
class SSR2HighResLoss(nn.Module):
    def __init__(self, loss_type, hr_state_dict_file, intensity_weight, hr_weights):
        super(SSR2HighResLoss, self).__init__()
        
        self.loss_type = loss_type
        self.hres = HighRes3DNet()
        self.hres.initLayer(hr_state_dict_file)
        self.hres.requires_grad = False # Freeze the highres3dnet
        self.hres.eval() # Set to eval mode to let the batchnorm layer works properly
        
        self.intensity_weight = intensity_weight
        self.hr_weights = hr_weights
        
        # A flag to control output format to fit for interfacing
        self.data_parallel = False
        
    def hrOutputs(self, x):
        output = []
        x = self.hres.conv_0(x)
        output.append(x)
        
        for res in self.hres.res_1:
            x = res(x)
        output.append(x)
        
        for res in self.hres.res_2:
            x = res(x)
        output.append(x)
        
        for res in self.hres.res_3:
            x = res(x)
        output.append(x)
        
        x = self.hres.conv_1(x)
        output.append(x)
        x = self.hres.conv_2(x)
        output.append(x)
        
        return output
    
    def forward(self, *inputs):
        prediction, ground_truth, mask = inputs
        
        # Intensity difference
        if self.loss_type == "l1":
#            intensity_loss = torch.abs((ground_truth - prediction)*mask).mean()
            intensity_loss = torch.abs(ground_truth - prediction).mean()
        elif self.loss_type == "l2":
            intensity_loss = (((ground_truth - prediction)*mask)**2).mean()
        
        gt_outputs = self.hrOutputs(ground_truth)
        pd_outputs = self.hrOutputs(prediction)
        
        # hr loss
        hrloss = 0
        for weight, gt_out, pd_out in zip(self.hr_weights, gt_outputs, pd_outputs):
            if self.loss_type == "l1":
#                hl = torch.abs((gt_out - pd_out)*mask).mean()
                hl = torch.abs(gt_out - pd_out).mean()
            elif self.loss_type == "l2":
                hl = (((gt_out - pd_out)*mask)**2).mean()
            
            hrloss += weight * hl
        
        # Total loss
        loss = intensity_loss + hrloss
        
        return loss

# This is for baseline comparison. Just implement the L1/L2 loss
class IntensityLoss(nn.Module):
    def __init__(self, loss_type):
        super(IntensityLoss, self).__init__()
        
        self.loss_type = loss_type
        # A flag to control output format to fit for interfacing
        self.data_parallel = False
        
    def forward(self, *inputs):
        prediction, ground_truth, mask = inputs
        
        # Intensity difference
        if self.loss_type == "l1":
#            intensity_loss = torch.abs((ground_truth - prediction)*mask).mean()
            intensity_loss = torch.abs(ground_truth - prediction).mean()
        elif self.loss_type == "l2":
#            intensity_loss = (((ground_truth - prediction)*mask)**2).mean()
            intensity_loss = ((ground_truth - prediction)**2).mean()
            
        return intensity_loss
        
# Ideally the discriminator should only need to be parallel apply
# TODO: implement the CTC loss
# https://github.com/Randl/improved-improved-wgan-pytorch/blob/master/models.py
#class SSR2DiscriminatorLoss(nn.Module):
#    
#    def __init__(self):
#        super(SSR2DiscriminatorLoss, self).__init__()
#        self.discriminator = SSR2Discriminator()
#        
#        self.grad_toggle = 'discriminator'
#        
#    def setGenerator(self):
#        self.grad_toggle = 'generator'
#        
#    def setDiscriminator(self):
#        self.grad_toggle = 'discriminator'
#    
#    def setDiscriminatorGrad(self):
#        self.grad_toggle = 'discriminator_grad'
#    
#    def forward(self, x, gt):
#        if self.grad_toggle == 'discriminator':
#            # In this case x should be pure real or fake image.
#            # The discriminator should produce 1 for real image, 0 for fake image
#            # The loss is between score and the ground truth label
#            scores = self.discriminator(x)
#            return ((scores - gt)**2).mean()
#        elif self.grad_toggle == 'discriminator_grad':
#            # In this case the input is linear interpolation of fake and real image.
#            # This is used to regularize the gradient of the discriminator. So all it
#            # returns is the mean of scores.
#            scores = self.discriminator(x)
#            return scores.mean()
#        elif self.grad_toggle == 'generator':
#            # In this case the x should be all fake image.
#            # Generator need to maximize the score getting from discriminator
#            # So the loss is the minus of the scores
#            scores = self.discriminator(x)
#            return -scores


# This is the function that calculates the gradient of discriminator on interpolation data
# Interpolation data is random interpolation between batch of real data and synthesized data.
# This forces the discriminator to have gradient within a given threshold, given input ranging from
# real data to synthesized data (thus the interpolation, to get data "in between"). This enforces 
# Lipschitz contiunity of the discriminator. 
def calc_gradient_penalty(netD, real_data, fake_data, data_parallel):
    # Sample random number for each sample in a batch
    bs, c, h, w, d = real_data.size()
    
    alpha = torch.rand(bs, 1, 1, 1, 1)
    alpha = alpha.expand(bs, c, h, w, d).contiguous()
    alpha = alpha.to(device = real_data.get_device())
    
    # Generate interpolation sample - these sample do not have to visually make sense. Just
    # a regularization to make sure the discriminator is smooth given whatever input
    interpolates = alpha * real_data.detach() + ((1.0 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True) # Default variable from detach has requires_grad_ disabled
    
    # Scatter the interpolates across gpus
    if data_parallel:
        interpolates_scatter = scatter(interpolates, netD.device_ids) 
        interpolates_scatter = tuple(interpolates_scatter)
        
        # wrap one more layer around
        # wrap one more layer around to fit discriminator input format
        interpolates_scatter = ( (interp,) for interp in interpolates_scatter ) 
        disc_interpolates = netD(interpolates_scatter).mean() # Pass through discriminator. The discriminator does the gather
    else:
        disc_interpolates = netD(interpolates).mean() # Pass through discriminator. The discriminator does the gather
        

    # The create_graph=True, retain_graph=True means constructing graph for the gradient operation
    # and allow higher order operation. We need to get gradient of gradient so these are set to true.
    # grad_outputs seems to be default value of gradient replacing None value if any. 
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device = disc_interpolates.get_device()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)  
    
    # It seems that here they arbitarily determine the discriminator should have gradient size of 1.                           
    # 1 as Lipschitz constant.
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty, interpolates

# Implement the CT loss for WGAN in 
# "IMPROVING THE IMPROVED TRAINING OF WASSERSTEIN GANS"
# Note that this does not implement the last conv part 
def consistency_term(real_data, discriminator, Mtag=0):
    d1 = discriminator(real_data)
    d2 = discriminator(real_data)

    # why max is needed when norm is positive?
    consistency_term = (d1 - d2).norm(2, dim=1).mean() - Mtag
    return consistency_term