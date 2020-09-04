# This contains utility that generate patches from 3D volume
#%% Import libraries
import numpy as np
import resample_util

#%% Utility functions


# This function sample 3D patches from 3D volume
# input:
#   image: the image in numpy array, dimension [H, W, D, C] 
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary
#   patch_size: the size of patch
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
# output:
#   patches_pos, patches_neg: list of (img_patch, seg_patch, cpt)
def single_resolution_patcher_3D(image, seg, patch_size, is_training = True, num_pos = 10, num_neg = 10, spacing = [1, 1, 1]):
    if is_training:
        # Randomly sample center points
        cpts_pos_sampled, cpts_neg_sampled = sample_center_points(seg, num_pos, num_neg)
        # Crop patches around center points
        patches_pos = crop_patch_by_cpts(image, seg, cpts_pos_sampled, patch_size)
        patches_neg = crop_patch_by_cpts(image, seg, cpts_neg_sampled, patch_size)
    
        return patches_pos, patches_neg
    else:
        # Regularly grid center points
        cpts = grid_center_points(image.shape, spacing)
        # Crop patches around center points
        patches = crop_patch_by_cpts(image, seg, cpts, patch_size)
        return patches
        

# This function sample 3D patches from 3D volume in multiple resolution around same center, used deepmedic or 
# similar style network
# input:
#   image: the image in numpy array, dimension [H, W, D, C] 
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary    
#   patchsize_multi_res: this is the patch size in multi-resolution [(1, (25, 25, 25)), (0.33, (19, 19, 19))]
#                   this means it will sample patch size (25, 25, 25) in resolution 1x, patch size (19, 19, 19) in resolution 0.33x etc    
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
def multi_resolution_patcher_3D(image, seg, patchsize_multi_res, is_training = True, num_pos = 10, num_neg = 10, spacing = [1, 1, 1]):
    # Sample center points
    if is_training:
        cpts_pos_sampled, cpts_neg_sampled = sample_center_points(seg, num_pos, num_neg)
        
        # Get center pts in multi resolution
        cpts_pos_multi_res = multiple_resolution_cpts(cpts_pos_sampled, patchsize_multi_res)
        cpts_neg_multi_res = multiple_resolution_cpts(cpts_neg_sampled, patchsize_multi_res)
        
        patches_pos_multi_res = []
        patches_neg_multi_res = []
        for idx, pr in enumerate(patchsize_multi_res):
            res, patch_size = pr
            # Downsample the image and segmentation
            image_resize, seg_resize = resample_util.resample_by_resolution(image, seg, res)
            
            cpts_max = np.array(image_resize.shape[:3]) - 1
            cpts_max = cpts_max[:, None]
            
            # Fetch positive patches
            cpts_pos = cpts_pos_multi_res[idx]
            cpts_pos = np.minimum(cpts_max, cpts_pos) # Limit the range
            # Due to numerical rounding the cpts in different resolution may not match the 
            # resize image exactly. So need to hard constraint it
            
            patches = crop_patch_by_cpts(image_resize, seg_resize, cpts_pos, patch_size)
            patches_pos_multi_res.append([patches, res])
            
            # Fetch positive patches
            cpts_neg = cpts_neg_multi_res[idx]
            cpts_neg = np.minimum(cpts_max, cpts_neg) # Limit the range.
            patches = crop_patch_by_cpts(image_resize, seg_resize, cpts_neg, patch_size)
            patches_neg_multi_res.append([patches, res])
        
        return patches_pos_multi_res, patches_neg_multi_res
    else:
        # Regularly grid center points
        cpts = grid_center_points(image.shape, spacing)
        cpts_multi_res = multiple_resolution_cpts(cpts, patchsize_multi_res)
        patches_multi_res = []
        
        for idx, pr in enumerate(patchsize_multi_res):
            res, patch_size = pr
            # Downsample the image and segmentation
            image_resize, seg_resize = resample_util.resample_by_resolution(image, seg, res)
            
            # Fetch patches
            cpts_res = cpts_multi_res[idx]
            patches_res = crop_patch_by_cpts(image_resize, seg_resize, cpts_res, patch_size)
            patches_multi_res.append([patches_res, res])
            
        return patches_multi_res

# This function samples center points from segmentation for patching. 
# Implement all patch selection in this function. Leave other function clean
# input:
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
def sample_center_points(seg, num_pos, num_neg):
    idx_pos = np.stack(np.where(seg>0), axis = 0)
    
    if idx_pos[0].shape[0]<num_pos:
        cpts_pos_sampled = idx_pos
    else:
        idx_rand = np.random.choice(idx_pos[0].shape[0], num_pos, replace = False)
        cpts_pos_sampled = idx_pos[:, idx_rand]
    
    if num_neg ==0:
        return cpts_pos_sampled, None

    idx_neg = np.stack(np.where(seg==0), axis = 0)
    
    if idx_neg[0].shape[0]<num_neg:
        cpts_neg_sampled = idx_neg
    else:
        idx_rand = np.random.choice(idx_neg[0].shape[0], num_neg, replace = False)
        cpts_neg_sampled = idx_neg[:, idx_rand]
    
    return cpts_pos_sampled, cpts_neg_sampled

# Inputs:
#   bbox: bbox of the mask, pick index from within
#   patch_size: the size of patch. 
# Output:
#   cpts: random center points sampled from it. 
def sample_center_points_by_bbox(bbox, patch_size, num_pts):
    half_size = np.array(patch_size) // 2
    
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    
    X, Y, Z = np.meshgrid(np.arange(xmin + half_size[0], xmax - half_size[0]), 
                np.arange(ymin + half_size[1], ymax - half_size[1]), 
                np.arange(zmin + half_size[2], zmax - half_size[2]))
    
    idx=np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis = 0)
    
    if idx[0].shape[0]<num_pts:
        cpts_sampled = idx
    else:
        idx_rand = np.random.choice(idx[0].shape[0], num_pts, replace = False)
        cpts_sampled = idx[:, idx_rand]
    
    return cpts_sampled

def grid_center_points_by_bbox(bbox, space, patch_size, include_end = True):
    half_size = np.array(patch_size) // 2
    
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
#    
#    xstart, xend = xmin + half_size[0], xmax - half_size[0]
#    ystart, yend = ymin + half_size[0], ymax - half_size[0]
#    zstart, zend = zmin + half_size[0], zmax - half_size[0]
    xstart, xend = xmin + half_size[0], xmax
    ystart, yend = ymin + half_size[1], ymax
    zstart, zend = zmin + half_size[2], zmax
    
    x, y, z = np.arange(xstart, xend, space[0]), np.arange(ystart, yend, space[1]), np.arange(zstart, zend, space[2])
    
    if include_end:
        if x[-1] + half_size[0] < xend:
            x = np.concatenate([x, [xend]])
        if y[-1] + half_size[1] < yend:
            y = np.concatenate([y, [yend]])
        if z[-1] + half_size[2] < zend:
            z = np.concatenate([z, [zend]])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")
    idx=np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis = 0)
    
    return idx

def grid_center_points_by_bbox_old(bbox, space, patch_size, include_end = True):
    half_size = np.array(patch_size) // 2
    
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    
    xstart, xend = xmin + half_size[0], xmax - half_size[0]
    ystart, yend = ymin + half_size[0], ymax - half_size[0]
    zstart, zend = zmin + half_size[0], zmax - half_size[0]
    
    x, y, z = np.arange(xstart, xend, space[0]), np.arange(ystart, yend, space[1]), np.arange(zstart, zend, space[2])
    
    if include_end:
        if x[-1] + half_size[0] < xend:
            x = np.concatenate([x, [xend]])
        if y[-1] + half_size[1] < yend:
            y = np.concatenate([y, [yend]])
        if z[-1] + half_size[2] < zend:
            z = np.concatenate([z, [zend]])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")
    idx=np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis = 0)
    
    return idx


# This function generate center points in order of image. Just to keep the API consistent
def grid_center_points(shape, space, include_end = True, patch_size = None):
    
    if patch_size is None:
        x = np.arange(start = 0, stop = shape[0], step = space[0])
        y = np.arange(start = 0, stop = shape[1], step = space[1])
        z = np.arange(start = 0, stop = shape[2], step = space[2])
        
        if include_end:
            x = np.concatenate([x, [shape[0] - 1]])
            y = np.concatenate([y, [shape[1] - 1]])
            z = np.concatenate([z, [shape[2] - 1]])
    else:
        # If patch size is given, then start with patch size half
        half_size = np.ceil((np.array(patch_size) - 1) / 2).astype(np.int32)
        x = np.arange(start = half_size[0], stop = shape[0], step = space[0])
        y = np.arange(start = half_size[1], stop = shape[1], step = space[1])
        z = np.arange(start = half_size[2], stop = shape[2], step = space[2])
        
        if include_end:
            if x[-1] + half_size[0] < shape[0]:
                x = np.concatenate([x, [shape[0] - 1]])
            if y[-1] + half_size[1] < shape[1]:
                y = np.concatenate([y, [shape[1] - 1]])
            if z[-1] + half_size[2] < shape[2]:
                z = np.concatenate([z, [shape[2] - 1]])
        
    x_t, y_t, z_t = np.meshgrid(x, y, z, indexing = "ij")
    idx = np.stack([x_t.flatten(), y_t.flatten(), z_t.flatten()], axis = 0)
    
    return idx
    
# This function converts center points to multiple resolution
def multiple_resolution_cpts(cpts, patchsize_multi_res):
    cpts_multi_res = []
    for pr in patchsize_multi_res:
        res, _ = pr
        cpts_res = (cpts * res).astype(np.int32)
        cpts_multi_res.append(cpts_res)
    return cpts_multi_res

# Sample function as above, except the input image and seg has already been padded. Also it only 
# input one cpt. This is used in a pytorch dataset.
def crop_single_patch(image_padded, seg_padded, cpt, patch_size):
    l = cpt
    u = cpt + np.array(patch_size)
    img_patch = image_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2], :]
    seg_patch = seg_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2]]
    return img_patch, seg_patch, cpt
        

# This function crops patches around center points with patch size
# input:
#   image: input image, [H, W, D, C]
#   seg: segmentation, [H, W, D]
#   cpts: center points, [3, N]
#   patch_size: tuple, (HP, WP, DP)
# output:
#   patches: list of (img_patch, seg_patch, cpt)
def crop_patch_by_cpts(image, seg, cpts, patch_size):
    half_size = np.ceil((np.array(patch_size) - 1) / 2).astype(np.int32)
    N = cpts.shape[1]
    patches = []
    
    # Padded the image and segmentation here so that the out of boundary cases are taken care of
    image_padded = np.pad(image, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2]), (0, 0)), mode = "constant", constant_values = 0)
    seg_padded = np.pad(seg, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2])), mode = "constant", constant_values = 0)
    
    shape = image.shape
    
    for i in range(N):
        cpt = cpts[:, i]
        l = cpt
        u = cpt + np.array(patch_size)
        img_patch = image_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2], :]
        seg_patch = seg_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2]]
        patches.append((img_patch, seg_patch, cpt, shape))
        
        if (img_patch.shape[0] == 95) or (img_patch.shape[1] == 95) or (img_patch.shape[2] == 95):
            print ("Debugging ", cpt, l, u, image_padded.shape)
        
    return patches

# This function assemble segmentation patches into - test version
# One can cross check with the above crop_patch_by_cpts and confirm that these 
# 2 does the exact reverse operations in terms of indexing. So patches cropped by
# the crop_patch_by_cpts when assembled by assemSegFromPatches will be correct.
# Input:
#   patches: list of n tuple, n being the number of patch this patient has. 
#            Each tuple of the following structure 
#            (ph[0], pl[0], seg, cpts, shape, disease, patient)
#            The only useful information here is just the cpts and shape
#   patches_pred: numpy array of shape [n, hs, ws, ds, C]. hs, ws, ds being size of the segmentation patch, 
#         C being probability of each label
# Output:
#   seg: numpy array of shape [H, W, D, C], H, W, D being size of complete image.
def assemSegFromPatches(shape, cpts, patches_pred, no_overlap = False):
    # Get shape param
    H, W, D = shape
    n, hs, ws, ds, C = patches_pred.shape
    
    im_shape = np.array([H, W, D])
    half_shape = np.ceil((np.array([hs, ws, ds]) - 1) / 2).astype(np.int32)
    padded_shape = (im_shape + half_shape * 2).astype(np.int32)
    
    # Pad the segmentation so that the edge cases are handled
    seg = np.zeros(list(padded_shape) + [C])
    rep = np.zeros(list(padded_shape) + [C])
    
    for cpt, pred in zip(list(cpts), list(patches_pred)):
        try:
            if no_overlap:
                patch_mask = rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :]
                patch_mask = (patch_mask==0).astype(np.float64) # only add to where that's empty
                seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += pred * patch_mask
                rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += patch_mask
            else:
                seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += pred
                rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += 1
        except ValueError:
            print ("Debug")
            print (cpt[0], cpt[0] + hs, cpt[1], cpt[1] + ws, cpt[2], cpt[2] + ds)
            print (pred.shape)
            print (seg.shape)
            print (seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :].shape)
            return None
            break
    
    # Crop out edges
    seg = seg[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
    rep = rep[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
    
    rep[rep==0] = 1e-6
    
    # Normalized by repetition
    seg = seg/rep
    
    return seg, rep


def initSegRep(shape, patch_size):
    H, W, D = shape
    im_shape = np.array([H, W, D])
    hs, ws, ds, C = patch_size
    half_shape = np.ceil((np.array([hs, ws, ds]) - 1) / 2).astype(np.int32)
    padded_shape = (im_shape + half_shape * 2).astype(np.int32)
    seg = np.zeros(list(padded_shape) + [C]).astype(np.float32)
    rep = np.zeros(list(padded_shape) + [C]).astype(np.float32)
    return seg, rep, half_shape
    
# This function given an semi finished segmentation continue to build it
def assemSegFromPatchesContinuous(seg, rep, shape, cpts, patches_pred, no_overlap = False):
    # Get shape param
    H, W, D = shape
    n, hs, ws, ds, C = patches_pred.shape
    
#    im_shape = np.array([H, W, D])
#    half_shape = np.ceil((np.array([hs, ws, ds]) - 1) / 2).astype(np.int32)
#    padded_shape = (im_shape + half_shape * 2).astype(np.int32)
#    
#    # Pad the segmentation so that the edge cases are handled
#    seg = np.zeros(list(padded_shape) + [C])
#    rep = np.zeros(list(padded_shape) + [C])
    
    for cpt, pred in zip(list(cpts), list(patches_pred)):
        try:
            if no_overlap:
                patch_mask = rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :]
                patch_mask = (patch_mask==0).astype(np.float32) # only add to where that's empty
                seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += pred.astype(np.float32) * patch_mask
                rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += patch_mask.astype(np.float32)
                del patch_mask
            else:
                seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += pred.astype(np.float32)
                rep[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += 1
        except ValueError:
            print ("Debug")
            print (cpt[0], cpt[0] + hs, cpt[1], cpt[1] + ws, cpt[2], cpt[2] + ds)
            print (pred.shape)
            print (seg.shape)
            print (seg[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :].shape)
            return None
            break
    
    # Crop out edges
#    seg = seg[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
#    rep = rep[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
#    rep[rep==0] = 1e-6
    
    # Normalized by repetition
#    seg = seg/rep
    
#    return seg, rep

def cropSeg(seg, rep, half_shape):
    seg = seg[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
    rep = rep[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
    rep[rep==0] = 1e-6
    
    seg = seg/rep
    
    return seg, rep
    