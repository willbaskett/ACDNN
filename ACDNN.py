######################################################################################################################################################
#ACDNN-CNNs are a specialized type of CNN which divides the network into hierarchical sub-networks of increasing complexity.
#Each sub-network produces a separate prediction. All predictions attempt to predict the same target, but have different numbers of parameters
#The network therefore learns to make predictions using many different levels of model complexity, all in a single model
######################################################################################################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import sys

def calculate_l2(model, device):
    l2 = None
    for param in model.parameters():
        if l2 is None:
            l2 = F.mse_loss(param, torch.zeros(param.shape, device=device), reduction="sum")
        else:
            l2 = l2 + F.mse_loss(param, torch.zeros(param.shape, device=device), reduction="sum")
    return l2

def restrict_info(x, is_training, std=0.01):
    if is_training:
        noise = torch.empty_like(x, device=x.device).normal_(std=std)
        x = x + noise
    return x

######################################################################################################################################################
#ACDNN1D
######################################################################################################################################################

class ResBlock1D(nn.Module):
    def __init__(self, n_hidden, n_out=None, group_size=1):
        super(ResBlock1D, self).__init__()

        if n_out is None:
            n_out = n_hidden
            self.end_block = False
        else:
            self.end_block = True
           
        self.norm1 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm1d(n_hidden, affine=False))
        self.fc1 = nn.Conv1d(n_hidden, n_hidden, 3)
        
        self.norm2 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm1d(n_hidden, affine=False))
        self.fc2 = nn.Conv1d(n_hidden, n_out, 3)
        
        mask1 = torch.eye(n_hidden//group_size).cumsum(0)
        mask1 = F.interpolate(mask1.reshape(1,1,mask1.shape[0], mask1.shape[1]), n_hidden).reshape(-1, n_hidden,1)
        self.register_buffer("mask1", mask1)

        mask2 = torch.eye(n_out//group_size).cumsum(0)
        mask2 = F.interpolate(mask2.reshape(1,1,mask2.shape[0], mask2.shape[1]), n_out).reshape(-1, n_out,1)[:,:n_hidden,:]
        self.register_buffer("mask2", mask2)
    
    def forward(self, x):
        x_in = x.clone()
        x_in_shape = x_in.shape
        x = self.norm1(x)
        x = F.conv1d(x, restrict_info(self.fc1.weight, self.training)*self.mask1, restrict_info(self.fc1.bias, self.training), padding="same")
        x = self.norm2(x)
        x = F.conv1d(x, restrict_info(self.fc2.weight, self.training)*self.mask2, restrict_info(self.fc2.bias, self.training), padding="same")
        x[:,:x_in_shape[1],:] += x_in
        if self.end_block and x.shape[2] > 1:
            x = F.max_pool1d(x,2)
            
        return x

class ResBlocks1D(nn.Module):
    def __init__(self, starting_channels, max_channels, in_features, out_features, n_blocks=1, blocks_per_level=1, group_size=1):
        super(ResBlocks1D, self).__init__()
        self.out_features = out_features
        
        if in_features is not None:
            self.features_in = nn.Conv2d(in_features, starting_channels, 1)
            
        current_channels = starting_channels
        new_channels = starting_channels
        
        self.blocks = []
        for i in range(n_blocks):
            n_out = None
            current_channels = new_channels
            if (i+1)%blocks_per_level == 0:
                if current_channels * 2 <= max_channels:
                    current_channels = new_channels
                    new_channels = new_channels * 2
                n_out = new_channels
            self.blocks.append(ResBlock1D(current_channels, n_out=n_out, group_size=group_size))
        
        self.current_channels = current_channels

        if out_features is not None:
            self.out_norm = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm1d(current_channels, affine=False))
            self.features_out = nn.Linear(current_channels, current_channels*out_features)
                               
        self.blocks = nn.ModuleList(self.blocks)
        
        mask = torch.eye((current_channels)//group_size).cumsum(0)
        mask = F.interpolate(mask.reshape(1,1,mask.shape[0], mask.shape[1]), current_channels).reshape(-1, current_channels)
        
        index_range = torch.arange(current_channels).repeat(out_features,1).T.reshape(-1)
        mask = mask[index_range,:]
        
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        
        if self.features_in is not None:
            x = F.conv1d(x, restrict_info(self.features_in.weight, self.training), restrict_info(self.features_in.bias, self.training), padding="same")

        for block in self.blocks:
            x = block(x)
        
        if self.features_out is not None:
            x = x.mean(dim=[2])
            x = self.out_norm(x)
            x = F.linear(x, restrict_info(self.features_out.weight, self.training)*self.mask, restrict_info(self.features_out.bias, self.training)).reshape(-1,self.current_channels, self.out_features).permute(0,2,1)
            
        return x

############################################################################################################################
#Accepts 1D inputs such as time-series sequences
#in_channels = number of channels in input
#starting_channels = number of channels in the first block of the ACDNN residual block, number of channels are doubled after each downsampling up to max_channels
#max_channels = the max number of allowed channels, see above
#n_out = number of outputs per head (number of classes)
#blocks = number of residual blocks
#blocks_per_level = number of blocks before a downsampling/channel doubling operation
#group size = number of nodes in each distinct path/sub-network
#random_input_mask = T/F if T randomly masks part of the input during training, useful for interpretation later
##########################################################################################################################
class ACDNN1D(nn.Module):
    def __init__(self, in_channels=7, starting_channels=8, max_channels=256, n_out=1, blocks=8, blocks_per_level=1, group_size=1, random_input_mask=False):
        super(ACDNN1D, self).__init__()
        self.group_size = group_size
        self.random_input_mask = random_input_mask
        
        self.encoder = ResBlocks1D(
                n_blocks=blocks,
                in_features=in_channels,
                out_features=n_out,
                starting_channels=starting_channels,
                max_channels=max_channels,
                group_size=group_size,
                blocks_per_level=blocks_per_level)

        self.n_outputs = self.encoder.current_channels//group_size

        self.bn = nn.BatchNorm1d(in_channels, affine=False, momentum=None)
        
    def forward(self, x, mask=None, use_random_mask=False):
        x = self.bn(x)
        if (self.training and self.random_input_mask) or use_random_mask or mask is not None:
            alphas = make_non_uniform_alphas(x)
            if mask is None:
                mask = alphas > torch.empty((alphas.shape[0],),device=x.device).uniform_(0,1).reshape(-1,1)
            x = x * mask.unsqueeze(1)
            
        x = self.encoder(x)

        #pool outputs within the same group
        x = F.avg_pool1d(x, kernel_size=self.group_size, stride=self.group_size)
        
        return x

######################################################################################################################################################
#ACDNN2D
######################################################################################################################################################

def make_non_uniform_alphas2d(x, smoothing_ratio=0.01, return_probability_dist=True):
    device = x.device
    B, C, H, W = x.shape
    pool_width = int(H*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, H, W), device=device)
    dimensions = H
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions), int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, (int(H), int(H)), mode="bilinear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier

    base = base.reshape(B, H, W)
    base = F.avg_pool2d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=[1,2]).reshape(-1,1,1)
    base = base / base.std(axis=[1,2]).reshape(-1,1,1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

class ResBlock2D(nn.Module):
    def __init__(self, n_hidden, n_out=None, group_size=1):
        super(ResBlock2D, self).__init__()

        if n_out is None:
            n_out = n_hidden
            self.end_block = False
        else:
            self.end_block = True
           
        self.norm1 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(n_hidden, affine=False))
        self.fc1 = nn.Conv2d(n_hidden, n_hidden, 3)
        
        self.norm2 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(n_hidden, affine=False))
        self.fc2 = nn.Conv2d(n_hidden, n_out, 3)
        
        mask1 = torch.eye(n_hidden//group_size).cumsum(0)
        mask1 = F.interpolate(mask1.reshape(1,1,mask1.shape[0], mask1.shape[1]), n_hidden).reshape(-1, n_hidden,1,1)
        self.register_buffer("mask1", mask1)

        mask2 = torch.eye(n_out//group_size).cumsum(0)
        mask2 = F.interpolate(mask2.reshape(1,1,mask2.shape[0], mask2.shape[1]), n_out).reshape(-1, n_out,1,1)[:,:n_hidden,:,:]
        self.register_buffer("mask2", mask2)
    
    def forward(self, x):
        x_in = x.clone()
        x_in_shape = x_in.shape
        x = self.norm1(x)
        x = F.conv2d(x, restrict_info(self.fc1.weight, self.training)*self.mask1, restrict_info(self.fc1.bias, self.training), padding="same")
        x = self.norm2(x)
        x = F.conv2d(x, restrict_info(self.fc2.weight, self.training)*self.mask2, restrict_info(self.fc2.bias, self.training), padding="same")
        x[:,:x_in_shape[1],:,:] += x_in
        if self.end_block and x.shape[2] > 1:
            x = F.max_pool2d(x,2)
            
        return x

class ResBlocks2D(nn.Module):
    def __init__(self, starting_channels, max_channels, in_features, out_features, n_blocks=1, blocks_per_level=1, group_size=1):
        super(ResBlocks2D, self).__init__()
        self.out_features = out_features
        
        if in_features is not None:
            self.features_in = nn.Conv2d(in_features, starting_channels, 1)
            
        current_channels = starting_channels
        new_channels = starting_channels
        
        self.blocks = []
        for i in range(n_blocks):
            n_out = None
            current_channels = new_channels
            if (i+1)%blocks_per_level == 0:
                if current_channels * 2 <= max_channels:
                    current_channels = new_channels
                    new_channels = new_channels * 2
                n_out = new_channels
            self.blocks.append(ResBlock2D(current_channels, n_out=n_out, group_size=group_size))
        
        self.current_channels = current_channels

        if out_features is not None:
            self.out_norm = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm1d(current_channels, affine=False))
            self.features_out = nn.Linear(current_channels, current_channels*out_features)
                               
        self.blocks = nn.ModuleList(self.blocks)
        
        mask = torch.eye((current_channels)//group_size).cumsum(0)
        mask = F.interpolate(mask.reshape(1,1,mask.shape[0], mask.shape[1]), current_channels).reshape(-1, current_channels)
        
        index_range = torch.arange(current_channels).repeat(out_features,1).T.reshape(-1)
        mask = mask[index_range,:]
        
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        
        if self.features_in is not None:
            x = F.conv2d(x, restrict_info(self.features_in.weight, self.training), restrict_info(self.features_in.bias, self.training), padding="same")

        for block in self.blocks:
            x = block(x)
        
        if self.features_out is not None:
            x = x.mean(dim=[2,3])
            x = self.out_norm(x)
            x = F.linear(x, restrict_info(self.features_out.weight, self.training)*self.mask, restrict_info(self.features_out.bias, self.training)).reshape(-1,self.current_channels, self.out_features).permute(0,2,1)
            
        return x

############################################################################################################################
#Accepts 2D inputs such as medical images (color images are also accepted)
#in_channels = number of channels in input
#starting_channels = number of channels in the first block of the ACDNN residual block, number of channels are doubled after each downsampling up to max_channels
#max_channels = the max number of allowed channels, see above
#n_out = number of outputs per head (number of classes)
#blocks = number of residual blocks
#blocks_per_level = number of blocks before a downsampling/channel doubling operation
#group size = number of nodes in each distinct path/sub-network
#random_input_mask = T/F if T randomly masks part of the input during training, useful for interpretation later
##########################################################################################################################
class ACDNN2D(nn.Module):
    def __init__(self, in_channels=7, starting_channels=8, max_channels=256, n_out=1, blocks=8, blocks_per_level=1, group_size=1, random_input_mask=False):
        super(ACDNN2D, self).__init__()
        self.group_size = group_size
        self.random_input_mask = random_input_mask
        
        self.encoder = ResBlocks2D(
                n_blocks=blocks,
                in_features=in_channels,
                out_features=n_out,
                starting_channels=starting_channels,
                max_channels=max_channels,
                group_size=group_size,
                blocks_per_level=blocks_per_level)

        self.n_outputs = self.encoder.current_channels//group_size

        self.bn = nn.BatchNorm2d(in_channels, affine=False, momentum=None)
        
    def forward(self, x, mask=None, use_random_mask=False):
        x = self.bn(x)
        if (self.training and self.random_input_mask) or use_random_mask or mask is not None:
            alphas = make_non_uniform_alphas2d(x)
            if mask is None:
                mask = alphas > torch.empty((alphas.shape[0],),device=x.device).uniform_(0,1).reshape(-1,1,1)
            x = x * mask.unsqueeze(1)
            
        x = self.encoder(x)

        #pool outputs within the same group
        x = F.avg_pool1d(x, kernel_size=self.group_size, stride=self.group_size)
        
        return x


######################################################################################################################################################
#ACDNN3D
######################################################################################################################################################

def make_non_uniform_alphas3d(x, smoothing_ratio=0.01, return_probability_dist=True):
    device = x.device
    B, C, H, W, D = x.shape
    pool_width = int(H*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, H, W, D), device=device)
    dimensions = H
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions), int(dimensions), int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, (int(H), int(H), int(H)), mode="trilinear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier

    base = base.reshape(B, H, W, D)
    base = F.avg_pool3d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=[1,2,3]).reshape(-1,1,1,1)
    base = base / base.std(axis=[1,2,3]).reshape(-1,1,1,1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

class ResBlock3D(nn.Module):
    def __init__(self, n_hidden, n_out=None, group_size=1):
        super(ResBlock3D, self).__init__()

        if n_out is None:
            n_out = n_hidden
            self.end_block = False
        else:
            self.end_block = True
           
        self.norm1 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm3d(n_hidden, affine=False))
        self.fc1 = nn.Conv3d(n_hidden, n_hidden, 3)
        
        self.norm2 = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm3d(n_hidden, affine=False))
        self.fc2 = nn.Conv3d(n_hidden, n_out, 3)
        
        mask1 = torch.eye(n_hidden//group_size).cumsum(0)
        mask1 = F.interpolate(mask1.reshape(1,1,mask1.shape[0], mask1.shape[1]), n_hidden).reshape(-1, n_hidden,1,1,1)
        self.register_buffer("mask1", mask1)

        mask2 = torch.eye(n_out//group_size).cumsum(0)
        mask2 = F.interpolate(mask2.reshape(1,1,mask2.shape[0], mask2.shape[1]), n_out).reshape(-1, n_out,1,1,1)[:,:n_hidden,:,:,:]
        self.register_buffer("mask2", mask2)
    
    def forward(self, x):
        x_in = x.clone()
        x_in_shape = x_in.shape
        x = self.norm1(x)
        x = F.conv3d(x, restrict_info(self.fc1.weight, self.training)*self.mask1, restrict_info(self.fc1.bias, self.training), padding="same")
        x = self.norm2(x)
        x = F.conv3d(x, restrict_info(self.fc2.weight, self.training)*self.mask2, restrict_info(self.fc2.bias, self.training), padding="same")
        x[:,:x_in_shape[1],:,:,:] += x_in
        if self.end_block and x.shape[2] > 1:
            x = F.max_pool3d(x,2)
            
        return x

class ResBlocks3D(nn.Module):
    def __init__(self, starting_channels, max_channels, in_features, out_features, n_blocks=1, group_size=1, blocks_per_level=1):
        super(ResBlocks3D, self).__init__()
        self.out_features = out_features
        
        if in_features is not None:
            self.features_in = nn.Conv3d(in_features, starting_channels, 1)
            
        current_channels = starting_channels
        new_channels = starting_channels
        
        self.blocks = []
        for i in range(n_blocks):
            n_out = None
            if (i+1)%blocks_per_level == 0:
                current_channels = new_channels
                if current_channels * 2 <= max_channels:
                    current_channels = new_channels
                    new_channels = new_channels * 2
                n_out = new_channels
            self.blocks.append(ResBlock3D(current_channels, n_out=n_out, group_size=group_size))
        
        self.current_channels = current_channels

        if out_features is not None:
            self.out_norm = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm1d(current_channels, affine=False))
            self.features_out = nn.Linear(current_channels, current_channels*out_features)
                               
        self.blocks = nn.ModuleList(self.blocks)
        
        mask = torch.eye((current_channels)//group_size).cumsum(0)
        mask = F.interpolate(mask.reshape(1,1,mask.shape[0], mask.shape[1]), current_channels).reshape(-1, current_channels)
        
        index_range = torch.arange(current_channels).repeat(out_features,1).T.reshape(-1)
        mask = mask[index_range,:]
        
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        
        if self.features_in is not None:
            x = F.conv3d(x, restrict_info(self.features_in.weight, self.training), restrict_info(self.features_in.bias, self.training), padding="same")

        for block in self.blocks:
            x = block(x)
        
        if self.features_out is not None:
            x = x.mean(dim=[2,3,4])
            x = self.out_norm(x)
            x = F.linear(x, restrict_info(self.features_out.weight, self.training)*self.mask, restrict_info(self.features_out.bias, self.training)).reshape(-1,self.current_channels, self.out_features).permute(0,2,1)
            
        return x

############################################################################################################################
#Accepts 3D inputs such as CT scans
#in_channels = number of channels in input
#starting_channels = number of channels in the first block of the ACDNN residual block, number of channels are doubled after each downsampling up to max_channels
#max_channels = the max number of allowed channels, see above
#n_out = number of outputs per head (number of classes)
#blocks = number of residual blocks
#blocks_per_level = number of blocks before a downsampling/channel doubling operation
#group size = number of nodes in each distinct path/sub-network
#random_input_mask = T/F if T randomly masks part of the input during training, useful for interpretation later
##########################################################################################################################
class ACDNN3D(nn.Module):
    def __init__(self, in_channels=7, starting_channels=8, max_channels=256, n_out=1, blocks=8, blocks_per_level=1, group_size=1, random_input_mask=False):
        super(ACDNN3D, self).__init__()
        self.group_size = group_size
        self.random_input_mask = random_input_mask
        
        self.encoder = ResBlocks3D(
                n_blocks=blocks,
                in_features=in_channels,
                out_features=n_out,
                starting_channels=starting_channels,
                max_channels=max_channels,
                group_size=group_size,
                blocks_per_level=blocks_per_level)

        self.n_outputs = self.encoder.current_channels//group_size

        self.bn = nn.BatchNorm3d(in_channels, affine=False, momentum=None)
        
    def forward(self, x, mask=None, use_random_mask=False):
        x = self.bn(x)
        if (self.training and self.random_input_mask) or use_random_mask or mask is not None:
            alphas = make_non_uniform_alphas3d(x)
            if mask is None:
                mask = alphas > torch.empty((alphas.shape[0],),device=x.device).uniform_(0,1).reshape(-1,1,1,1)
            x = x * mask.unsqueeze(1)
            
        x = self.encoder(x)

        #pool outputs within the same group
        x = F.avg_pool1d(x, kernel_size=self.group_size, stride=self.group_size)
        
        return x