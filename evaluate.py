# Training script written by Jiancong Wang. 
# This file is annotated for debug purpose

from __future__ import print_function
from __future__ import absolute_import
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

#from sync_batchnorm.replicate import patch_replication_callback

import os
import shutil

# Network definition:
from tqdm import tqdm

from attrdict import AttrDict
import nibabel as nib
from MRIDatasetEval import create_split, load_split, ImageDataset, PatchDataset
from RRDBGenerator import define_G_64_nobn
from patch_util import assemSegFromPatches, grid_center_points_by_bbox

experiment_config = {}
experiment_config['LOW_RES_DIR'] = "./lowres"
experiment_config['HIGH_RES_DIR'] = "./highres"
experiment_config['MASK_CACHE_DIR'] = "./mask_cache"

experiment_config['OUTPUT_DIR'] = "./evalout"
experiment_config['PATCH_SIZE'] = (64, 40, 64)
experiment_config['MARGIN'] = 3
experiment_config['SPACING'] = (
        experiment_config['PATCH_SIZE'][0] - 2*experiment_config['MARGIN'], 
        experiment_config['PATCH_SIZE'][1] - 2*experiment_config['MARGIN'], 
        experiment_config['PATCH_SIZE'][2] - 2*experiment_config['MARGIN']
        )

# Setting batchsize and data parallel - check out, both single and multi-gpu works
experiment_config['BATCH_SIZE'] = 1
experiment_config["DATA_PARALLEL"] = False
device_ids = (3,)
gpu_id = device_ids[0] # for debug purpose

# Generator model
experiment_config['G_model'] = "RRDB_G64_nobn"
experiment_config['checkname'] = experiment_config['G_model']

# This function given a model name returns the model
def G_model_selector(G_model):
    if G_model == "RRDB_G64_nobn":
        return define_G_64_nobn()
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
        
        test_list = [f for f in os.listdir(experiment_config['LOW_RES_DIR']) if f.endswith(".nii.gz")]
        test_dataset = ImageDataset(test_list, self.config.HIGH_RES_DIR, self.config.LOW_RES_DIR, self.config.MASK_CACHE_DIR, istrain = False, whitening = True, mask_type = 'mean_plus', augment_type = [])
        
        # Note that here the batch_size must be 1 and there is actually no reason to use batchsize that's larger.
        print("Building image data loader")
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)
        
        # Define model 
        print("Building model")
        
        # Build generator
        self.generator = G_model_selector(self.config.G_model)
        
        # Using cuda
        print("Porting model and loss to CUDA")
        if args.cuda:
            self.generator = self.generator.cuda(device_ids[0])
            # When using data parallel, the model must located at gpu of device_ids[0]
            if self.config.DATA_PARALLEL:
                self.generator = nn.DataParallel(self.generator, device_ids = device_ids )
                
        print("Done porting to CUDA")
            
        # If needed, one can adjust different learning rate for different part of the model
        # here. Refers to https://pytorch.org/docs/stable/optim.html
        
        # Put the right value of weight decay here
        # Load checkpoint
        self.best_pred = 0.0
        print("Loading checkpoint")
        self.epoch = 0
        self.global_step = 0
        
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(self.args.resume, map_location='cuda:0')
            self.args.start_epoch = checkpoint['epoch'] # Starting epoch
            
            # Model parameter
            if args.cuda and self.config.DATA_PARALLEL:
                self.generator.module.load_state_dict(checkpoint['G_state_dict'], strict= (self.config.G_model != "mDC_stage2"))
            else:
                self.generator.load_state_dict(checkpoint['G_state_dict'], strict= (self.config.G_model != "mDC_stage2"))
            
            # Training stat
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            raise ValueError("No valid checkpoint found!")
            
    def validation(self, split = 'validation'):
        # Set model to eval mode
        self.generator.train() 
        
        loader = self.test_loader
        # Set generator to eval mode, so that dropout will not activate
        # Also note that the batchnorm will become eval mode here
        tbar = tqdm(loader, desc='\r')
        
        # output directory for reconstruction
        output_dir = os.path.join(self.config.OUTPUT_DIR, self.config.checkname, split)
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Loop through all images
        for i, sample in enumerate(tbar):
            # Loop through patches in images
            image = np.stack([sample['image_lr'].numpy(), sample['image_hr'].numpy()], axis = -1)
            image = np.squeeze(image)
            mask = np.squeeze(sample['mask'].numpy())
            bbox = (0, 178, 0, 223, 0, 223 ) # This covers the whole image
            
            patdset = PatchDataset(image, mask, bbox,
                                   istrain = False, patch_size = self.config.PATCH_SIZE, spacing = self.config.SPACING)
            patloader = data.DataLoader(patdset, batch_size = self.config.BATCH_SIZE, shuffle = False, 
                                        drop_last = False, pin_memory = True, num_workers = 4)

            # Loop through patches in images
            patches = []
            cpts = []
            for psample in patloader:
                cpts.append(np.squeeze(psample['cpt'].numpy()))
                patch_lr = psample['patch_lr']
                patch_hr = psample['patch_hr']

                if self.args.cuda:
                    patch_lr = patch_lr.cuda(device = gpu_id)
                    patch_hr = patch_hr.cuda(device = gpu_id)
                
                with torch.no_grad():
                    patch_sr = self.generator(patch_lr)
                    
                    
                    patches.append(patch_sr.detach().cpu().numpy())
                    
            # This should release all the releasble memory 
            torch.cuda.empty_cache() 
            
            # Assemble the patches into full volume
            patches = np.concatenate(patches)
            cpts = np.concatenate(cpts) if self.config.DATA_PARALLEL else np.stack(cpts)
            patches = np.transpose(patches, axes = [0, 2, 3, 4, 1])
            margin = self.config.MARGIN
            patches = patches[:, margin:-margin, margin:-margin, margin:-margin, :]
            shape = image.shape[:3] # Image shape
            img_sr, rep = assemSegFromPatches(shape, cpts, patches, no_overlap = True)
            
            # Add back the mean and std
            mean_hr = np.squeeze(sample['mean_hr'].numpy())
            std_hr = np.squeeze(sample['std_hr'].numpy())
            img_sr = img_sr * std_hr + mean_hr
            
            # Save the assembled image out
            im_file = sample['im_file'][0]
            affine = sample['affine'].numpy()[0]
            nib.save(nib.Nifti1Image(img_sr, affine), os.path.join(output_dir, im_file) )
            
def main(args):
    # Build trainer object
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    print("Done Building trainer object")
    
    # Now the epoch and step are stored in the trainer
    trainer.validation(split = 'test')
    
if  __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate SR Model')
    parser.add_argument('--no_cuda',action='store_true', help='Dont use gpu for training.')
    # The order of the config file must be default,yamk before the celeba-10pts.yaml, because 
    # the metayaml must parse the depending variable in default before can parse celeba.
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, 
                        default="./RRDB_G64_nobn/experiment_1/checkpoint.pth.tar",
                        help='put the path to resuming file if needed')
    # evaluation option
    args = parser.parse_args()
    main(args)
