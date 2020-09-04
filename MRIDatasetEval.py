#%% This class is dataloader for the MRI data

import os
import numpy as np
import numpy.ma as ma
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib

from masking import make_mask_3d, get_mask_bbox_3D
from sklearn.model_selection import train_test_split
import patch_util

import csv

# The mean and variance normalization
def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    mean = masked_img.mean()
    std = masked_img.std()
    image = (image - mean) / max(std, 1e-5)
    
#    image = (image - 916.9705865132373) / 555.1283378783712
    return image, mean, std


# Create and store simple split
# 780 training, 111 validation, 111 evaluation and 111 test samples by subject    
    
def create_split(imagelist, split_dir):
    train, val_eval_test = train_test_split(imagelist, train_size = 780)
    val, eval_test = train_test_split(val_eval_test, train_size = 111)
    evaluation, test = train_test_split(eval_test, train_size = 111)
    
    with open(split_dir, 'w') as f:
        writer = csv.writer(f)
        for im in train:
            writer.writerow([im, 'train'])
        for im in val:
            writer.writerow([im, 'val'])
        for im in evaluation:
            writer.writerow([im, 'evaluation'])
        for im in test:
            writer.writerow([im, 'test'])
            
    return train, val, evaluation, test


def load_split(split_dir):
    train, val, evaluation, test = [], [], [], []
    with open(split_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == 'train':
                train.append(row[0])
            elif row[1] == 'val':
                val.append(row[0])
            elif row[1] == 'evaluation':
                evaluation.append(row[0])
            elif row[1] == 'test':
                test.append(row[0])
            else:
                raise ValueError("Unknown split")
    return train, val, evaluation, test



# Function fetched and adapted from this thread
# https://stackoverflow.com/questions/43922198/how-to-rotate-a-3d-image-by-a-random-angle-in-python
def random_rotation_3d(image, mask, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    A randomly rotated 3D image and the mask
    """
    # Consider this function being used in multithreading in pytorch's dataloader,
    # if one don't reseed each time this thing is run, the couple worker in pytorch's
    # data worker will produce exactly the same random number and that's no good.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    image_raw = image.copy()
    # rotate along z-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(0, 1), reshape=False, order = 3)
#    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(0, 1), reshape=False, order = 0)
    mask = scipy.ndimage.interpolation.rotate(image_raw, angle, mode='nearest', axes=(0, 1), reshape=False, order = 3)
     
    # rotate along y-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(0, 2), reshape=False, order = 3)
#    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(0, 2), reshape=False, order = 0)
    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='nearest', axes=(0, 2), reshape=False, order = 3)

    # rotate along x-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(1, 2), reshape=False, order = 3)
#    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(1, 2), reshape=False, order = 0)
    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='nearest', axes=(1, 2), reshape=False, order = 3)

    return image, mask

# This dataset loads whole image and its binary mask, does image level normalization and augmentation
class ImageDataset(Dataset):
    # imagelist: directories of all the nii brain images
    def __init__(self, imagelist, high_res_dir, low_res_dir, mask_cache_dir, istrain = True, whitening = True, mask_type = 'mean_plus', augment_type = ['flip', 'rotate'], 
                 augment_param = {'max_angle' : 10}
                 ):
        self.imagelist = imagelist
        self.istrain = istrain
        self.mask_cache_dir = mask_cache_dir
        self.mask_type = mask_type
        self.augment_type = augment_type
        self.augment_param = augment_param
        self.whitening = whitening
        
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, idx):
        # Image level normalization and augmentation put here
        im_file = self.imagelist[idx]
        
        # Low resolution image
        img = nib.load(os.path.join(self.low_res_dir, im_file))
        affine = img.affine # The affine matrix that convert the pixel space to world space
        image_lr = img.get_fdata()
        image_lr = np.squeeze(image_lr)
        
        img = nib.load(os.path.join(self.high_res_dir, im_file))
        affine = img.affine # The affine matrix that convert the pixel space to world space
        image_hr = img.get_fdata()
        image_hr = np.squeeze(image_hr)
        
        
        # Making mask is a time consuming process since it needs binary fill holes
        # cache it to disk once generated
        mask_dir = os.path.join(self.mask_cache_dir, im_file.split("/")[-1])
        if os.path.exists(mask_dir):
            mask = nib.load(mask_dir).get_fdata().astype(np.uint8)
            bbox = np.load(mask_dir.replace('.nii.gz', '.npy'))
        else:
            mask = make_mask_3d(image_lr, type_str = self.mask_type).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), mask_dir)
            # bbox of the mask
            bbox = get_mask_bbox_3D(mask)
            np.save(mask_dir.replace('.nii.gz', '.npy'), bbox)
            
        # Augmentation
        # The augmentation is inspired by lfz/deeplung's data.py
        # If train, do augmentation. If test, no augmentation is needed
        if self.istrain:
            # left right flip
            if 'flip' in self.augment_type:
                flip = np.random.random() > 0.5
                if flip:
                    image_lr = image_lr[:, :, ::-1]
                    image_hr = image_hr[:, :, ::-1]
                    mask = mask[:, :, ::-1]
            
            # Random rotate
            if 'rotate' in self.augment_type:
                rotate = np.random.random() > 0.5
                if rotate:
                    images = np.stack([image_lr, image_hr], axis = -1)
                    images, mask = random_rotation_3d(images, mask, self.augment_param['max_angle'])
                    image_lr = image_lr[:, :, :, 0]
                    image_hr = image_hr[:, :, :, 1]
                    
        # Mean and var normalization
        if self.whitening:
            image_lr, mean_lr, std_lr = whitening_transformation(image_lr, mask)
            image_hr, mean_hr, std_hr = whitening_transformation(image_hr, mask)
                    
        return {'image_lr': torch.from_numpy(image_lr.astype(np.float32)), 
                'image_hr': torch.from_numpy(image_hr.astype(np.float32)), 
                'mask' : torch.from_numpy(np.ascontiguousarray(mask)), 
                'mask_bbox' : torch.from_numpy(bbox), 
                'affine' : torch.from_numpy(affine),
                'idx' : idx,
                'im_file': self.imagelist[idx],
                'mean_lr': mean_lr, 
                'std_lr': std_lr,
                'mean_hr': mean_hr, 
                'std_hr': std_hr
                }
            
# This dataset given a full image, fetch patch from it
class PatchDataset(Dataset):
    def __init__(self, image, mask, mask_bbox, patch_size, istrain = False, spacing = [64, 40, 64], num_pos = 10):
#        print ("Image shape: ", image.shape)
#        print ("Mask shape: ", mask.shape)
        
        assert np.all(np.array(image.shape[:-1]) == np.array(mask.shape))
        
        self.istrain = istrain
        self.patch_size = patch_size
        self.mask_bbox = mask_bbox
        self.spacing = spacing
        self.data_shape = image.shape
        self.num_pos = num_pos
        
        # Note that the combo of np.ceil and -1 here is used to handle when the patch_size is even.
        half_size = np.ceil((np.array(patch_size) - 1) / 2).astype(np.int32)
        
        # Expand image dimension if only single channel
        if image.ndim == 3:
            image = image[...,None]

        # Padded the image and segmentation here so that the out of boundary cases are taken care of
        self.image_padded = np.pad(image, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2]), (0, 0)), mode = "constant", constant_values = 0)
        self.mask_padded = np.pad(mask, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2])), mode = "constant", constant_values = 0)
        self.half_size = half_size
    
        if self.istrain:
            # For training
            # Randomly sample center points
            self.cpts_sampled = patch_util.sample_center_points_by_bbox(mask_bbox, patch_size, num_pos)
        else:
            # For testing
            # Regularly sampled grid center points
#            self.cpts_sampled = patch_util.grid_center_points(image.shape, spacing, patch_size = self.patch_size )
#            self.cpts_sampled = patch_util.grid_center_points_by_bbox_old(self.mask_bbox, self.spacing, self.patch_size, include_end = True)
            self.cpts_sampled = patch_util.grid_center_points_by_bbox(self.mask_bbox, self.spacing, self.spacing, include_end = True)
        
    def __len__(self):
        return self.cpts_sampled.shape[1]
    
    def __getitem__(self, idx):
        cpt = self.cpts_sampled[:, idx]
        img_patch, mask_patch, cpt = patch_util.crop_single_patch(self.image_padded, 
                                                                 self.mask_padded,
                                                                 cpt, 
                                                                 self.patch_size)
        # Fit pytorch dimensions
        # [H, W, D, C] -> [C, H, W, D]
        img_patch = np.transpose(img_patch, axes = [3, 0, 1, 2])
        mask_patch = mask_patch[None, :]
        
        return {'patch_lr' : torch.from_numpy(img_patch[0:1]),
                'patch_hr' : torch.from_numpy(img_patch[1:2]),
                'mask': torch.from_numpy(mask_patch.astype(np.float32)), # add type cast here 
                'cpt': torch.from_numpy(cpt)}
