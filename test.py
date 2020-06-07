import torch 
import torch.nn as nn
import torch.nn.functional as F 
from resnet import Network
from torch import optim
from dataset import PngDataset
from torch.utils import data
from utils.metrics.meter import AverageMeter
import time
from losses import MatrixLoss
import utils.data.data_io as io
import glob
import os
import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN

torch.set_default_tensor_type(torch.FloatTensor)
model = "/media/liu/large_disk/advanced_machine_learning/model/similarity_modelat_epoch_005.dat"
config = {}
batch_size = 3
config["input_shape"] = (batch_size*6,3,768,768)
config["base_channels"] = 16
config["depth"]=47
config["block_type"]="bottleneck"
net = Network(config)
net = io.load_model(net, model, load_weights=True)
net.to("cuda: 0")
glob_path = "./val_jpg_1/*"
for file in glob.iglob(glob_path):
    vector_list = []
    for folder in os.listdir(file):
        png_path = os.path.join(file,folder)
        png_array = cv2.imread(png_path).transpose(2,0,1)[np.newaxis,:,:,:]
        png_torch = torch.from_numpy(png_array).float()
        png_torch = png_torch.to("cuda: 0")
        vector = net(png_torch)
        vector = vector.detach().cpu().numpy()
        vector_list.append(vector)
    vector_total_array = np.stack(vector_list,axis=0)
    print (vector_total_array.shape)
    vector_total_array_part = vector_total_array.transpose(1,0,2)
    distance_array = (vector_total_array-vector_total_array_part)**2
    distance_array = np.sum(distance_array,axis=2)
    print (distance_array)


    vector_total_array = vector_total_array.reshape(vector_total_array.shape[0],vector_total_array.shape[2])
    db = DBSCAN(eps=0.05, min_samples=1).fit(vector_total_array)
    label = db.labels_
    print (label)

        

        # png_torch = torch.

