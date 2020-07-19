import random 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils import data
import json
import os
import cv2
import torch.nn as nn
import torch

class MatrixLoss(nn.Module):
    def __init__(self,interval):
        super(MatrixLoss, self).__init__()
        self.interval = interval
    
    def forward(self,input,target):
        input = input.unsqueeze(dim=1)
        input_part1 = input.permute(1,0,2)*-1
        input_result = (input+(input_part1*-1))**2
        input_result = input_result.sum(2)
        target = target.cuda()
        target_part1 = target.permute(1,0)
        target_result = target+(target_part1*-1)
        # print (target_result == 0)
        same_target = (target_result == 0)
        diff_target = (target_result != 0)
        same_distance = (same_target*input_result).sum(1)
        diff_distance = (diff_target*input_result).sum(1)
        same_target_mask = same_target.sum(1)
        diff_target_mask = diff_target.sum(1)
        same_distance = same_distance/same_target_mask 
        diff_distance = diff_distance/diff_target_mask
        #print ("same: ",same_distance)
        #print ("diff: ",diff_distance)
        margin_distance = (self.interval+same_distance-diff_distance)
        margin_mask = margin_distance>0
        margin_loss = (margin_distance*margin_mask).sum()
        if margin_mask.sum()>0:
            total_loss = same_distance.mean()+margin_loss/margin_mask.sum()
        else:
            total_loss = same_distance.mean()
        return total_loss



