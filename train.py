# Training script written by Jiancong Wang. 
# This file is annotated for debug purpose

from __future__ import print_function
from __future__ import absolute_import
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import scatter, gather
import torch.utils.data as data
from torch.utils.data import DataLoader

from sync_batchnorm.replicate import patch_replication_callback

import os
import shutil
import pdb

# Network definition:
from tqdm import tqdm
from saver import Saver
from collections import OrderedDict

from attrdict import AttrDict
from generator_torch import SSR2Generator
from discriminator_torch import SSR2Discriminator
from loss import SSR2HighResLoss, calc_gradient_penalty, consistency_term, IntensityLoss

from MRIDataset import create_split, load_split, ImageDataset, PatchDataset
from timeit import default_timer as timer

import nibabel as nib

from tensorboardX import SummaryWriter

from FSRCNN_torch import FSRCNN, SRResiNet


# This trainer class is inspired by the semantic segmentation training framework. 
# from https://github.com/speedinghzl/Pytorch-Deeplab
# Put the right value of weight decay in the optimizer
# Load config - Done
# Adapt the model to use sync batchnorm in the generator
# Define model
# Define loss
# Parallel model
# Parallel loss - need to interface the loss input and the model output
#   - Normalize the input and generated image
#   - fix the tuple sequence
# Restore model
# Exponential decay learning rate
# Build dataset
# Run training/validation 
#   interface the dataloader output -
#   Subtlety here - need to save the validation loss (reconstruction loss) minimum model - Done
# Save model
# Implement gradient clipping
# Interface the generated image to highres3dnet - need to normalize
# Change the main function - Done

experiment_config = {}
experiment_config['LOG_DIR'] = "/home/picsl/Desktop/SRBrain/log"
experiment_config['SPLIT_DIR'] = "/home/picsl/Desktop/SRBrain/split.csv"
experiment_config['LOW_RES_DIR'] = "/media/picsl/Storage2/HumanConnectome_lowres"
experiment_config['HIGH_RES_DIR'] = "/media/picsl/Storage2/HumanConnectome_highres"
experiment_config['OUTPUT_DIR'] = "/home/picsl/Desktop/SRBrain/output"
experiment_config['TENSORBOARD_DIR'] = "/home/picsl/Desktop/SRBrain/tfboard"
experiment_config['MASK_CACHE_DIR'] = "/home/picsl/Desktop/SRBrain/mask_cache"
experiment_config['hr_state_dict_file'] = "/home/picsl/Desktop/SRBrain/highres3dnet_params.pt"

experiment_config['loss_type'] = "l1"

experiment_config['WEIGHT_DECAY'] = 1e-4 # This seems to be reasonable value

experiment_config['PATCH_SIZE'] = (64, 40, 64)
experiment_config['NUM_PATCH'] = 48 # number of patch to get per image

# Generator loss weight
experiment_config['intensity_weight'] = 1
#experiment_config['hr_weights'] = np.array([1, 0.01, 0.01]) * 0.1
experiment_config['hr_weights'] = np.array([1, 0.01, 0.01, 0.01, 1, 0.5]) * 5
experiment_config['Use_hres_loss'] = True

experiment_config['D_weight_schedule'] = {
        (0, 2): 1e-1,
        (2, 10): 1e-1,
        (10, None): 1e-1,
        }

# This function is used to add various D weight
def D_weight_schedular(epoch, D_weight_schedule):
    for interval, dw in D_weight_schedule.items():
        epmin, epmax = interval
        epmax = 10000000 if epmax is None else epmax
        if epoch >= epmin and epoch < epmax:
            return dw

# Discriminator setting
experiment_config['D_patch_gan'] = True
experiment_config['D_dropout'] = 0.2 # Dropout rate when training D

# Discriminator loss weight
experiment_config['grad_penalty_weight'] = 10 # Some people use 10. It seems that the discriminator 
#experiment_config['CT_loss_weight'] = 2 # Some people use this number

# will output very large number. It is the gradient penalty that keeps it from exploding. So this 
# regularization has to be heavy.
# This is a 6 value array, corresponding to weighting of the value output by the 
# highres3dnet [conv_0, res_1, res_2, res_3, conv_1, conv_2]

# Discriminator loss weight
experiment_config['grad_weights'] = 1 # loss for weight of the gradient

experiment_config['G_ITER'] = 1
experiment_config['D_ITER'] = 5

# D's learning rate is larger than G's
#experiment_config['G_LEARNING_RATE'] = 5e-6
#experiment_config['D_LEARNING_RATE'] = 5e-5
#experiment_config['G_LEARNING_RATE'] = 5e-5
experiment_config['G_LEARNING_RATE'] = 1e-6
experiment_config['D_LEARNING_RATE'] = 5e-6
experiment_config["LR_DECAY"] = 0.95 # 
experiment_config["LR_STEP"] = 10000 # 10k
experiment_config["NUM_STEPS"] = 1000000 # 1M

# Setting batchsize and data parallel - check out, both single and multi-gpu works
experiment_config['BATCH_SIZE'] = 4
experiment_config["DATA_PARALLEL"] = True

experiment_config['D_ahead_epoch'] = 2 # Train the discriminator for 2 epoch before the generator comes in

device_ids = (0, 1, 2, 3)
gpu_id = device_ids[1] # for debug purpose



# Using only 15 images without D, the G overfits to the intensity loss at epoch 7. So we know roughly
# this or even smaller epoch we can add the D in.

# On the other hand, the D given only 15 training images and 5 val images overfits significantly. fake-real 
# even become positive. So maybe patch gan is a good idea.Since we try to recover details, this kind of makes 
# sense.

# Generator model
# Right now, 3 model to choose from
# FSRCNN, SRResiNet, mDCSRN
# TODO: implement mDPN, STARGAN and etc
experiment_config['G_model'] = "mDCSRN"
experiment_config['checkname'] = experiment_config['G_model'] + "_hres6_5_D01_2ahead"

# This function given a model name returns the model
def G_model_selector(G_model):
    if G_model == "mDCSRN":
        return SSR2Generator()
    elif G_model == "FSRCNN":
        return FSRCNN()
    elif G_model == "SRResiNet":
        return SRResiNet()
    else:
        raise ValueError("Unknown generator model")

class Trainer():
    def __init__(self, args):
        # Save args
        self.args = args
        
        # Load config files and parameters
        print ('Loading config')
        self.config = AttrDict(experiment_config)
        
        # Build train and test dataset -
        # default datasets parameters
        print("Building Dataset")
        
        # load split
        if os.path.exists(self.config.SPLIT_DIR):
            train_list, val_list, evaluation_list, test_list = load_split(self.config.SPLIT_DIR)
        else:
            imagelist = [ f for f in os.listdir(self.config.LOW_RES_DIR) if f.endswith(".nii.gz")]
            train_list, val_list, evaluation_list, test_list = create_split(imagelist, self.config.SPLIT_DIR)
        
        # Debug: shorten the train_list to just see run through one time.
        train_list = train_list[:70]
        val_list = val_list[:10]
        train_dataset = ImageDataset(train_list, self.config.HIGH_RES_DIR, self.config.LOW_RES_DIR, self.config.MASK_CACHE_DIR, istrain = True, whitening = True, mask_type = 'mean_plus', augment_type = [])
        val_dataset = ImageDataset(val_list, self.config.HIGH_RES_DIR, self.config.LOW_RES_DIR, self.config.MASK_CACHE_DIR, istrain = False, whitening = True, mask_type = 'mean_plus', augment_type = [])
        
        # Call the dataloader maker
        # Note that train_config.dset is the large dataset. celeba and aflw.
        # The smaller subset is in the config train_dset_params/test_dset_params "dataset"
        print("Building image data loader")
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=6)
        
        # Define model 
        print("Building model")
        
        
        # Build generator
        self.generator = G_model_selector(self.config.G_model)
        
        # Since there is layer norm in the discriminator, it needs to know input size.
        self.discriminator = SSR2Discriminator(input_size = self.config.PATCH_SIZE,
                                               patch_gan = self.config.D_patch_gan, 
                                               dropout_rate = self.config.D_dropout
                                               )
        
        # Define loss
        print("Building loss")
        
        if self.config.Use_hres_loss:
            self.highresloss = SSR2HighResLoss(self.config.loss_type, self.config.hr_state_dict_file, self.config.hr_weights)
            
        self.intloss = IntensityLoss(self.config.loss_type)
        
        # Using cuda
        print("Porting model and loss to CUDA")
        if args.cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.discriminator = self.discriminator.cuda(device_ids[0])
            
            if self.config.Use_hres_loss:
                self.highresloss = self.highresloss.cuda(device_ids[0])
            self.intloss = self.intloss.cuda(device_ids[0])
            # When using data parallel, the model must located at gpu of device_ids[0]
            if self.config.DATA_PARALLEL:
                self.generator = nn.DataParallel(self.generator, device_ids = device_ids )
                self.discriminator = nn.DataParallel(self.discriminator, device_ids = device_ids )
                
                # Set output device - even out memory consumption
                self.generator.output_device = gpu_id
                self.discriminator.output_device = gpu_id
                
                if self.config.Use_hres_loss:
                    self.highresloss = nn.DataParallel(self.highresloss, device_ids = device_ids)
                    self.highresloss.output_device = gpu_id
                
        print("Done porting to CUDA")
            
        # If needed, one can adjust different learning rate for different part of the model
        # here. Refers to https://pytorch.org/docs/stable/optim.html
        generator_params = [
                {'params': self.generator.parameters()},
        ]
        discriminator_params = [
                {'params': self.discriminator.parameters()},
        ]
        
        # Define optimizer and weight decay for generator and discriminator
        # Put the right value of weight decay here
        print("Building optimizer")
        G_lr = self.config.G_LEARNING_RATE
        D_lr = self.config.D_LEARNING_RATE
#        self.G_optimizer = torch.optim.Adam(generator_params, lr=G_lr, weight_decay=self.config.WEIGHT_DECAY) 
#        self.D_optimizer = torch.optim.Adam(discriminator_params, lr=D_lr, weight_decay=self.config.WEIGHT_DECAY) 
        self.G_optimizer = torch.optim.Adam(generator_params, lr=G_lr) 
        self.D_optimizer = torch.optim.Adam(discriminator_params, lr=D_lr) 
        
        # Define saver object
        if self.config.checkname is None:
            self.config.checkname = "SRBrain"
        self.saver = Saver(self.config.LOG_DIR, "HumanConnectome", self.config.checkname)
        self.saver.save_experiment_config(experiment_config) # Save experiment setup
        
        # Load checkpoint
        self.best_pred = 0.0
        print("Loading checkpoint")
        
        self.epoch = 0
        self.global_step = 0
        self.D_ahead_epoch = self.config.D_ahead_epoch
        
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(self.args.resume)
            
            # Model parameter
            # Decide whether this is normal resume or loaded from pretrained without discriminator
            if 'D_state_dict' in checkpoint and 'D_optimizer' in checkpoint:
                if args.cuda and self.config.DATA_PARALLEL:
                    self.generator.module.load_state_dict(checkpoint['G_state_dict'])
                    self.discriminator.module.load_state_dict(checkpoint['D_state_dict'])
                else:
                    self.generator.load_state_dict(checkpoint['G_state_dict'])
                    self.discriminator.load_state_dict(checkpoint['D_state_dict'])
                
                # Optimizer parameter
                self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
                self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
                
                # D ahead epoch
#                self.D_ahead_epoch = checkpoint['D_ahead_epoch']
                self.D_ahead_epoch = 2
            else:
                # Just load the generator. No other things
                if args.cuda and self.config.DATA_PARALLEL:
                    self.generator.module.load_state_dict(checkpoint['G_state_dict'])
                else:
                    self.generator.load_state_dict(checkpoint['G_state_dict'])
            # Training stat
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
                
        # Tensorboard X
        self.writer = SummaryWriter(log_dir=self.config.TENSORBOARD_DIR)
        
        
    def training(self):
        # Set model to train mode
        self.generator.train()
        self.discriminator.train()
        
        # Loss parameters for display
        G_int_loss = 0.0
        G_hr_loss = 0.0
        G_Dloss = 0.0
        generator_iter = 0.0
        
        discriminator_loss = 0.0
        discriminator_grad = 0.0
        discriminator_iter = 0.0
        
        tbar = tqdm(self.train_loader)
        
        # Loop through all images
        for i, sample in enumerate(tbar):
            # Loop through patches in images
            image = np.stack([sample['image_lr'].numpy(), sample['image_hr'].numpy()], axis = -1)
            image = np.squeeze(image)
            mask = np.squeeze(sample['mask'].numpy())
            bbox = np.squeeze(sample['mask_bbox'].numpy())
            
            patdset = PatchDataset(image, mask, bbox,
                                   istrain = True, patch_size = self.config.PATCH_SIZE, num_pos = self.config.NUM_PATCH)
            patloader_torch = data.DataLoader(patdset, batch_size = self.config.BATCH_SIZE, shuffle = True, 
                                        drop_last = True, pin_memory = True, num_workers = 4)

            patloader = iter(patloader_torch)
            patch_left = True
            circle_once = False
            
            while(patch_left):
                G_run = False
                D_run = False
            
                # Update global step - implement the exponenetial learning rate
                # Timer
#                start = timer()
                #---------------------TRAIN D------------------------
                if self.config.DATA_PARALLEL:
                    self.discriminator.module.setDropout('test') # Disable dropout
                else:
                    self.discriminator.setDropout('test') # Disable dropout
                # Train discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                # Loss for tensorboard display
                D_fake_minus_real_iter = 0
                
                for i in range(self.config.D_ITER):
#                    print("training D")
                    patch = next(patloader, None)
                    if patch is None:
                        patch_left = False # 
                        
                        if not circle_once:
                            patloader = iter(patloader_torch)
                            patch = next(patloader, None)
                        else:
                            break
                        # Reinit the patch loader for now. Run through the remaining steps.
                    
                    D_run = True # Indicator that D has run at least once
                    
                    patch_lr = patch['patch_lr']
                    patch_hr = patch['patch_hr']
                    mask = patch['mask']
                    
                    if self.args.cuda:
                        patch_lr = patch_lr.cuda(device = gpu_id)
                        patch_hr = patch_hr.cuda(device = gpu_id)
                        mask = mask.cuda(device = gpu_id)
                    
                    # Gen fake data and load real data
                    # Generate fake data
                    with torch.no_grad():
                        patch_sr = self.generator(patch_lr)
                        del patch_lr
                    
                    patch_sr = patch_sr.detach()
                    patch_sr.requires_grad = True
                    
                    # train with fake data - the sr data is already scatter so should be direct input
                    disc_fake = self.discriminator(patch_sr)
                    disc_fake = disc_fake.mean()
                        
                    # If data parallel, scatter the high resolution patches onto multiple gpu
                    # If data parallel, the discriminator will received scatter inputs.
                    disc_real = self.discriminator(patch_hr)
                    disc_real = disc_real.mean()
                    
                    # Consistent loss
#                    disc_real2 = self.discriminator(patch_hr)
#                    ct_loss = (disc_real - disc_real2).norm(2, dim=1).mean()
                    
                    # train with interpolates data
                    gradient_penalty, _ = calc_gradient_penalty(self.discriminator, patch_hr, patch_sr, data_parallel = False)
        
                    # final disc cost
                    # Want to score on fake data to be low, on real data high and want gradient 
                    # closed to Lipschitz constant
                    disc_cost = disc_fake - disc_real + self.config.grad_penalty_weight * gradient_penalty 
#                    + self.config.CT_loss_weight * ct_loss
                    
                    # zero grad-backward-step
                    torch.cuda.empty_cache() 
                    self.D_optimizer.zero_grad()
                    disc_cost.backward()
                    self.D_optimizer.step()
                    
                    # Update loss for display
                    w_dist = disc_fake - disc_real
                    discriminator_loss += w_dist.detach().cpu().numpy()
                    discriminator_grad += gradient_penalty.detach().cpu().numpy()
                    discriminator_iter += 1
                    
                    # For tensorboard display
                    D_fake_minus_real_iter += w_dist.detach().cpu().numpy()
                    
                    # This should release all memory
                    del patch_hr, mask, patch_sr, disc_cost, disc_fake, disc_real, w_dist
                    torch.cuda.empty_cache()
                
                #---------------------TRAIN G------------------------
               
                
                # Loss for tensorboard display
                G_int_loss_iter = 0.0
                G_hr_loss_iter = 0.0
                G_Dloss_iter = 0.0
                
                if self.config.DATA_PARALLEL:
                    self.discriminator.module.setDropout('test') # Set the dropout to be 0
                else:
                    self.discriminator.setDropout('test') # Set the dropout to be 0
                # Freeze discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                
                for i in range(self.config.G_ITER):
                    # Only train G when D is ahead for certain epoch.
                    if self.D_ahead_epoch >0:
                        continue
                
#                    print("training G")
                    patch = next(patloader, None)
                    if patch is None:
                        patch_left = False
                        
                        if not circle_once:
                            patloader = iter(patloader_torch)
                            patch = next(patloader, None)
                        else:
                            break
                        # Reinit the patch loader for now. Run through the remaining steps.
                    
                    G_run = True # Indicator that G has run at least once
                    
                    patch_lr = patch['patch_lr']
                    patch_hr = patch['patch_hr']
                    mask = patch['mask']
    
                    if self.args.cuda:
                        patch_lr = patch_lr.cuda(device = gpu_id)
                        patch_hr = patch_hr.cuda(device = gpu_id)
                        mask = mask.cuda(device = gpu_id)
                    
                    patch_sr = self.generator(patch_lr) 
                    # Intensity loss and highres3dnet response loss
                    int_loss = self.intloss(patch_sr, patch_hr, mask)
                    # Want to max the score of discriminator. Therefore as loss add minus sign here
                    D_loss = -self.discriminator(patch_sr).mean() # Pass the synthesized image through discriminator
                    loss = int_loss + D_loss * D_weight_schedular(self.epoch, self.config.D_weight_schedule)
                    
                    if self.config.Use_hres_loss:
                        hr_loss = self.highresloss(patch_sr, patch_hr, mask).mean()
                        loss += hr_loss
                    
                    # Debug - for debug purpose, first not use the discriminator for generator loss
#                    loss = hr_loss 
                    # zero grad-backward-step
                    torch.cuda.empty_cache() 
                    self.G_optimizer.zero_grad()
                    loss.backward()
                    self.G_optimizer.step()
                    
                    # Update loss for display
                    G_int_loss += int_loss.detach().cpu().numpy()
                    
                    if self.config.Use_hres_loss:
                        G_hr_loss += hr_loss.detach().cpu().numpy()
                    
                    G_Dloss += D_loss.detach().cpu().numpy()
                    generator_iter += 1
                    
                    # For Tensorboard display
                    G_int_loss_iter += int_loss.detach().cpu().numpy()
                    
                    if self.config.Use_hres_loss:
                        G_hr_loss_iter += hr_loss.detach().cpu().numpy()
                    
                    G_Dloss_iter += D_loss.detach().cpu().numpy()
                    
                    # This should release all the releasble memory 
                    del patch_lr, patch_hr, mask, patch_sr, D_loss, loss 
                    torch.cuda.empty_cache() 
                    
                # Write the loss to tensorboard X
                
                if G_run:
                    self.writer.add_scalar('G_int_loss_train', G_int_loss_iter/self.config.G_ITER, self.global_step)
                    self.writer.add_scalar('G_hr_loss_train', G_hr_loss_iter/self.config.G_ITER, self.global_step)
                    self.writer.add_scalar('G_Dloss_train',G_Dloss_iter/self.config.G_ITER, self.global_step)
                
                # Write the loss to tensorboard X
                if D_run:
                    self.writer.add_scalar('D_fake_minus_real_train', D_fake_minus_real_iter/self.config.D_ITER, self.global_step)
                
                # Indicator an image has been used by both the D and G
                circle_once = True 
                
                if D_run and G_run:
                    self.global_step += 1 # A global step only counts when one cycle of D and G both finished
                
                # Timer
#                end = timer()
#                print("Batch time: ", end - start)
            # Display the loss in the progress bar
            if generator_iter>0 and discriminator_iter>0:
                tbar.set_description('G loss: %.3f, %.3f, %.3f, D loss:  %.3f, %.3f ' % 
                                     (G_int_loss/generator_iter, G_hr_loss/generator_iter, G_Dloss/generator_iter, 
                                      discriminator_loss/discriminator_iter, discriminator_grad/discriminator_iter))
            elif generator_iter==0 and discriminator_iter>0:
                tbar.set_description('D loss:  %.3f, %.3f ' % 
                                     (discriminator_loss/discriminator_iter, discriminator_grad/discriminator_iter))
                

    def validation(self):
        # Set model to eval mode
        self.generator.eval() # Set generator to eval mode, so that dropout will not activate
        self.discriminator.eval() # Set generator to eval mode, so that dropout will not activate
        tbar = tqdm(self.val_loader, desc='\r')
        
        # output directory for display
        epoch_dir = os.path.join(self.config.OUTPUT_DIR, "epoch" + str(self.epoch))
        
        if os.path.exists(epoch_dir):
            shutil.rmtree(epoch_dir)
        
        os.makedirs(epoch_dir)
        
        # Loss parameters for display
        G_int_loss = 0.0
        G_hr_loss = 0.0
        G_Dloss = 0.0
        generator_iter = 0.0
        
        discriminator_loss = 0.0
        discriminator_iter = 0.0
        
        # Loop through all images
        
        for i, sample in enumerate(tbar):
            # Loop through patches in images
            image = np.stack([sample['image_lr'].numpy(), sample['image_hr'].numpy()], axis = -1)
            image = np.squeeze(image)
            mask = np.squeeze(sample['mask'].numpy())
            bbox = np.squeeze(sample['mask_bbox'].numpy())
            
            patdset = PatchDataset(image, mask, bbox,
                                   istrain = False, patch_size = self.config.PATCH_SIZE, spacing = self.config.PATCH_SIZE)
            patloader = data.DataLoader(patdset, batch_size = self.config.BATCH_SIZE, shuffle = False, 
                                        drop_last = True, pin_memory = True, num_workers = 4)

            patloader = iter(patloader)
            patch_left = True
            
            # Loop through patches in images
            counter = 0
            while(patch_left):
                patch = next(patloader, None)
                if patch is None:
                    patch_left = False
                    break
                patch_lr = patch['patch_lr']
                patch_hr = patch['patch_hr']
                mask = patch['mask']

                if self.args.cuda:
                    patch_lr = patch_lr.cuda(device = gpu_id)
                    patch_hr = patch_hr.cuda(device = gpu_id)
                    mask = mask.cuda(device = gpu_id)
                
                with torch.no_grad():
                    # If data parallel, the generator will generate patches scattered around gpus
                    # and the discriminator will produce gathered input. 
                    patch_sr = self.generator(patch_lr)
                    # Intensity loss and highres3dnet response loss
                    
                    if self.config.Use_hres_loss:
                        hr_loss = self.highresloss(patch_sr, patch_hr, mask).mean()
                    int_loss = self.intloss(patch_sr, patch_hr, mask)
                    # Want to max the score of discriminator. Therefore as loss add minus sign here
                    disc_fake = self.discriminator(patch_sr).mean() # Pass the synthesized image through discriminator
                    
                    # If data parallel, the discriminator will received scatter inputs.
                    disc_real = self.discriminator(patch_hr)
                    disc_real = disc_real.mean()
                    
                    # Update loss for display
                    # G
                    G_int_loss += int_loss.detach().cpu().numpy()
                    
                    if self.config.Use_hres_loss:
                        G_hr_loss += hr_loss.detach().cpu().numpy()
                    G_Dloss += -disc_fake.detach().cpu().numpy()
                    
                    generator_iter += 1
                    
                    # D
                    w_dist = disc_fake - disc_real
                    discriminator_loss += w_dist.detach().cpu().numpy()
                    discriminator_iter += 1
            
                # This should release all the releasble memory 
                torch.cuda.empty_cache() 
                
                # For every couple images, save couple patches for display
                # Debug - output more image and more more patches to check
                if i % 10 == 0 and counter % 2 == 0:
                    idx = 0 if self.config.BATCH_SIZE == 1 else 1
                    patch_lr = np.squeeze(patch['patch_lr'][idx].numpy())
                    patch_hr = np.squeeze(patch['patch_hr'][idx].numpy())
                    patch_mask = np.squeeze(patch['mask'][idx].numpy())
                    
                    patch_sr = np.squeeze( patch_sr[idx].detach().cpu().numpy())
                    self.save_patches(epoch_dir, patch_lr, patch_hr, patch_sr, patch_mask,  i, counter)
                
                counter += 1
                    
            # Display the loss in the progress bar
            if generator_iter>0 and discriminator_iter>0:
                tbar.set_description('G loss: %.3f, %.3f, %.3f, D loss:  %.3f' % 
                                     (G_int_loss/generator_iter, G_hr_loss/generator_iter, G_Dloss/generator_iter, 
                                      discriminator_loss/discriminator_iter))
            
        # Write the loss to tensorboard X
        if generator_iter>0:
            self.writer.add_scalar('G_int_loss_val', G_int_loss/generator_iter, self.global_step)
            self.writer.add_scalar('G_hr_val', G_hr_loss/generator_iter, self.global_step)
            self.writer.add_scalar('G_Dloss_val', G_Dloss/generator_iter, self.global_step)
        if discriminator_iter>0:
            self.writer.add_scalar('D_fake_minus_real_val', discriminator_loss/discriminator_iter, self.global_step)
            # If the discriminator's performance start to degrade compared to the generator, train the D more to compensate.
            if discriminator_loss/discriminator_iter > -0.5:
                self.D_ahead_epoch = 3 # Set 3 here so outside -1 will become 2
        
        self.D_ahead_epoch = max(self.D_ahead_epoch - 1, 0)
        self.epoch = self.epoch + 1 # Update epochs
        
        # Save checkpoints
        is_best = True
        parallel_model = self.args.cuda and self.config.DATA_PARALLEL
        get_dict = lambda model: model.module.state_dict() if parallel_model else model.state_dict()
        self.saver.save_checkpoint({
            'epoch': self.epoch,
            'D_ahead_epoch' : self.D_ahead_epoch,
            'global_step': self.global_step,
            'G_state_dict': get_dict(self.generator),
            'D_state_dict':  get_dict(self.discriminator),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict(),
            'best_pred': G_int_loss/generator_iter
        }, is_best)
    
    
    # This function save feature maps for visualization
    def saveFeatures(self):
        # Set model to eval mode
        self.generator.eval() # Set generator to eval mode, so that dropout will not activate
        tbar = tqdm(self.val_loader, desc='\r')
        
        # output directory for display
        epoch_dir = os.path.join(self.config.OUTPUT_DIR, "epoch_feat_" + str(self.epoch))
        if os.path.exists(epoch_dir):
            shutil.rmtree(epoch_dir)
        
        os.makedirs(epoch_dir)
        
        # Loss parameters for display
        for i, sample in enumerate(tbar):
            # Loop through patches in images
            image = np.stack([sample['image_lr'].numpy(), sample['image_hr'].numpy()], axis = -1)
            image = np.squeeze(image)
            mask = np.squeeze(sample['mask'].numpy())
            bbox = np.squeeze(sample['mask_bbox'].numpy())
            
            patdset = PatchDataset(image, mask, bbox,
                                   istrain = False, patch_size = self.config.PATCH_SIZE, spacing = self.config.PATCH_SIZE)
            patloader = data.DataLoader(patdset, batch_size = self.config.BATCH_SIZE, shuffle = False, 
                                        drop_last = True, pin_memory = True, num_workers = 4)

            patloader = iter(patloader)
            patch_left = True
            
            # Loop through patches in images
            counter = 0
            while(patch_left):
                patch = next(patloader, None)
                if patch is None:
                    patch_left = False
                    break
                patch_lr = patch['patch_lr']
                patch_hr = patch['patch_hr']
                mask = patch['mask']

                if self.args.cuda:
                    patch_lr = patch_lr.cuda(device = gpu_id)
                    patch_hr = patch_hr.cuda(device = gpu_id)
                    mask = mask.cuda(device = gpu_id)
                
                with torch.no_grad():
                    # If data parallel, the generator will generate patches scattered around gpus
                    # and the discriminator will produce gathered input. 
                    patch_sr = self.generator(patch_lr)
                    torch.cuda.empty_cache() 
                    feat_list = self.generator.module.get_features(patch_lr.to(device_ids[0]))
                    
                # This should release all the releasble memory 
                torch.cuda.empty_cache() 
                
                # For every couple images, save couple patches for display
                # Debug - output more image and more more patches to check
                if i % 10 == 0 and counter % 2 == 0:
                    idx = 0 if self.config.BATCH_SIZE == 1 else 1
                    patch_lr = np.squeeze(patch['patch_lr'][idx].numpy())
                    patch_hr = np.squeeze(patch['patch_hr'][idx].numpy())
                    patch_mask = np.squeeze(patch['mask'][idx].numpy())
                    
                    patch_sr = np.squeeze( patch_sr[idx].detach().cpu().numpy())
                    self.save_patches(epoch_dir, patch_lr, patch_hr, patch_sr, patch_mask,  i, counter)
                    self.save_feat(epoch_dir, idx, feat_list, i, counter)
                
                counter += 1
                    
    # This function saves patches for display
    def save_patches(self, output_dir, patch_lr, patch_hr, patch_sr, patch_mask, image_idx, patch_idx):
        lr = nib.Nifti1Image(patch_lr, np.eye(4))
        hr = nib.Nifti1Image(patch_hr, np.eye(4))
        sr = nib.Nifti1Image(patch_sr, np.eye(4))
        mask = nib.Nifti1Image(patch_mask, np.eye(4))
        
        nib.save(lr, os.path.join(output_dir, '%d_%d_lr.nii.gz'% (image_idx, patch_idx)))
        nib.save(hr, os.path.join(output_dir, '%d_%d_hr.nii.gz'% (image_idx, patch_idx)))
        nib.save(sr, os.path.join(output_dir, '%d_%d_sr.nii.gz'% (image_idx, patch_idx)))
        nib.save(mask, os.path.join(output_dir, '%d_%d_mask.nii.gz'% (image_idx, patch_idx)))
    
    # This function saves features for inspection
    def save_feat(self, output_dir, idx, features, image_idx, patch_idx):
        for feat_idx, feat in enumerate(features):
            feat_np = feat[idx].detach().cpu().numpy()
            feat_np = np.transpose(feat_np, axes = [1, 2, 3, 0]) # Shift the feature dimension to last
            feat_nib = nib.Nifti1Image(feat_np, np.eye(4))
            nib.save(feat_nib, os.path.join(output_dir, '%d_%d_feat_%d.nii.gz'% (image_idx, patch_idx, feat_idx)))
        
    # This function implements the exponential decay learning rate
    def exponential_decay(self, global_step, 
                          learning_rate = 0.001, decay_rate = 0.95, decay_steps = 100000,
                          staircase = True):
        if staircase:
            m = int(global_step / decay_steps)
        else:
            m = global_step / decay_steps
        decayed_learning_rate = learning_rate * decay_rate**m
        return decayed_learning_rate
        
    # This function given an optimizer and learning rate, adjust the optimizer's 
    # learning rate
    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # manipulate the lr at different group
            for i in range(0, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

def main(args):
    # Build trainer object
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    print("Done Building trainer object")
    print('Starting steps:', trainer.epoch * len(trainer.train_loader))
    print('Total steps:', trainer.config.NUM_STEPS)
    
    # Option for saving feature map for inspection
    if args.save_feat:
        trainer.saveFeatures()
        return
    
    # Now the epoch and step are stored in the trainer
    while True:
        trainer.training()
        trainer.validation()
        if trainer.global_step>=trainer.config.NUM_STEPS:
            break
    
    trainer.writer.close() # Guess this will never be run. But for completeness.
    
if  __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train SR Model')
    parser.add_argument('--no_cuda',action='store_true', help='Dont use gpu for training.')
    # The order of the config file must be default,yamk before the celeba-10pts.yaml, because 
    # the metayaml must parse the depending variable in default before can parse celeba.
    
    parser.add_argument('--save_feat', type = bool, default=True, 
                        help='This option runs the network on validation data and save the output feature for inspection' )
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, 
#                        default="/home/picsl/Desktop/SRBrain/log/HumanConnectome/mDCSRN_hres6_5/experiment_1/checkpoint.pth.tar",
                        default="/home/picsl/Desktop/SRBrain/log/HumanConnectome/mDCSRN_hres6_5_D01_2ahead/experiment_3/checkpoint.pth.tar",
#                        default=None,
                        help='put the path to resuming file if needed')
    # evaluation option
    args = parser.parse_args()
    main(args)
