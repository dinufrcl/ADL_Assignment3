# # allowed libraries
# import torch
# import argparse
# import random
# import time
# import torchvision
# from torch.cuda.amp import GradScaler
# from torch.utils.data import DataLoader
# from dataset import SkijumpDataset,load_dataset
# from torch.utils.tensorboard import SummaryWriter
# from eval import test
# from torch import autocast
# import datetime
# import PIL
# import numpy as np
# import timm
# import cv2
# import matplotlib
# import os
# import sys
# import tqdm
# from model import BaseResNetModel,HRNet
# # avoid for loops and list-maps operation
# #The only way to omit for-loops is to use vectorized operations with numpy or torch
#
#
# def from_heatmaps_to_keypoints(heatmap: torch.Tensor, scaling_factor_heatmap: torch.Tensor,scaling_factor, bounding_box: torch.Tensor, threshhold):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     original_flatten = heatmap.view(heatmap.shape[0], heatmap.shape[1], -1)
#     max_value, max_ind = original_flatten.max(-1)
#     max_activation = torch.stack([torch.div(max_ind,heatmap.shape[3],rounding_mode='trunc'), max_ind % heatmap.shape[3]], -1)
#
#     scaling_factor_heatmap = scaling_factor_heatmap.view(-1,1,1)
#     max_activation = max_activation / scaling_factor_heatmap
#     max_activation = max_activation[:,:,[1,0]]
#     scaling_factor = scaling_factor.view(-1,1,1)
#     max_activation = max_activation /scaling_factor
#     max_activation[:,:,0] += bounding_box[:,3].view(-1,1)
#     max_activation[:, :, 1] += bounding_box[:, 1].view(-1,1)
#     max_activation = torch.round(max_activation)
#     visible = torch.ones((max_activation.shape[0],17,1)) *2
#     visible[max_value <= threshhold] = 0
#     visible = visible.to(device)
#     max_activation = torch.concat((max_activation,visible),dim=2).view(-1,51)
#
#
#     return max_activation
#
#
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Human Pose Estimation')
#     parser.add_argument('-m', '--model', type=str, default="resnet", choices=['resnet','hrnet'],
#                         help='Specify the model')
#     parser.add_argument('-e', '--epochs', type=int, default=100, help="Epochs for the training")
#     parser.add_argument('-bs', '--batchsize', type=int, default=16, help="Batch size for the training")
#     parser.add_argument('-aug', '--augmentation', type=bool, default=False, help='Data augmentation for the training')
#     parser.add_argument('-w', '--worker', type=int, default=0, help="Worker for the training")
#
#     args = parser.parse_args()
#
#     return args
#
#
# # task 1e
# def train(args):
#     """
#     training of the model
#     :param args: parse args
#     :return:
#     """
#     epochs = args.epochs
#     bs = args.batchsize
#     model = args.model
#     augmentation = args.augmentation
#     worker = args.worker
#
#     print(f'Epochs: {epochs}')
#     print(f'Batch size: {bs}')
#     print(f'Model type: {model}')
#     print(f'Augmentation: {augmentation}')
#     print(f'Worker: {worker}')
#
#     torch.manual_seed(79)
#     random.seed(79)
#     np.random.seed(79)
#     start_time = time.time()
#     min_val = np.inf
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     if model == 'hrnet':
#         model = HRNet()
#     else:
#         model = BaseResNetModel()
#
#     optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)
#     criterion = torch.nn.MSELoss()
#     scaler = GradScaler()
#
#     x, y, z = load_dataset('data/dataset/annotations/train.csv', 'data/dataset/annotated_frames')
#     train_dataloader = DataLoader(SkijumpDataset(x,y,z,128,augmentation), batch_size=bs, shuffle=True, drop_last=True,num_workers=worker)
#     x, y, z = load_dataset('data/dataset/annotations/train.csv', 'data/dataset/annotated_frames')
#     val_dataloader = DataLoader(SkijumpDataset(x,y,z,128,False), batch_size=bs,num_workers=worker)
#
#     model.train()
#     model = model.to(device)
#     writer = SummaryWriter(
#         log_dir='data/graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") + "" + model.class.name_)
#
#     print("Starting Training")
#     for epoch in range(epochs):
#
#         model.train()
#         epoch_loss = 0.0
#         for i, data in enumerate(train_dataloader, 0):
#
#             inputs, kp, heatmaps, sfh, bounding, sf, kpo  = data
#             inputs = inputs.to(device)
#             heatmaps = heatmaps.to(device)
#             optim.zero_grad()
#
#             with autocast(device_type='cuda', dtype=torch.float16):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, heatmaps)
#
#             scaler.scale(loss).backward()
#             scaler.step(optim)
#             scaler.update()
#             epoch_loss += loss.item()
#
#
#         epoch_loss = epoch_loss / len(train_dataloader)
#         writer.add_scalar('Training/Loss', epoch_loss, epoch + 1)
#         print(f'Epoch {epoch + 1} out of {epochs}')
#         print("Train loss: " + str(epoch_loss))
#         min_val = test(val_dataloader, model, writer, criterion, epoch,min_val)
#         print('---------------------')
#
#
# if _name_ == "_main_":
#     args = parse_args()
#     train(args)