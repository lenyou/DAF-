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


if __name__ == "__main__":
    end_epoch = 30
    lr_rate = 1e-2
    config = {}
    batch_size = 3
    config["input_shape"] = (batch_size*6,3,768,768)
    config["base_channels"] = 16
    config["depth"]=47
    config["block_type"]="bottleneck"
    net = Network(config)
    net = nn.DataParallel(net).cuda()
    matrixloss = MatrixLoss(interval=2)
    milestones = [8,15]
    lr_gamma = 0.2
    start_lr = 1e-3
    weight_decay = 2e-3
   
    batch_size = 5
    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)
    png_dataset = PngDataset("./train_jpg_1")
    data_loader = data.DataLoader(dataset=png_dataset, batch_size=batch_size, shuffle=True,drop_last = True, num_workers=4)
    
    for epoch in range(end_epoch):
        train_scheduler.step()
        batch_time_val = AverageMeter()
        loss_val = AverageMeter()
        end = time.time()
        show_count = 0
        for batch_idx, data in enumerate(data_loader):
            input, target = data
            input = input.view(-1,input.size()[2],input.size()[3],input.size()[4]).float()
            target = target.view(-1,1).long()
            vector = net(input)
            loss = matrixloss(vector,target)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            loss_val.update(loss.mean().item())
            res = '\t'.join(['Train', 'Loss: ', '%f'%(loss.mean()),"batch_idx:","%d"%batch_idx])
            if batch_idx % 5 == 0:
                print(res)
        print ("end epoch saving model")
        io.save_model(net, epoch,"./model","similarity_model")

