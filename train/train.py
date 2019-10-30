from datetime import datetime
import glob
import os
import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from model import *

class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepaths, target_filepaths, map_filepaths=None):
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.map_filepaths = map_filepaths
        
        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)        
        assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        target_filepath = self.target_filepaths[idx]        
        sample_token = input_filepath.split("/")[-1].replace("_input.png","")        
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)        
        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)        
        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)        
        im = im.astype(np.float32)/255
        target = target.astype(np.int64)        
        im = torch.from_numpy(im.transpose(2,0,1))
        target = torch.from_numpy(target)      
        return im, target, sample_token

if __name__=="__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu",action='store_true', help="whether use multi_gpu")
    parser.add_argument("--ckpt", type=str, default="ckpt/", help="path to store checkpoint file")
    parser.add_argument("--data_folder", type=str, default="data/lyft_bev", help="path to load bev data")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size to load data")
    parser.add_argument("--reslayers", type=int, default=34, help="choose resnet layers from [18, 34, 50]")
    parser.add_argument("--epochs", type=int, default=20, help="Training Epochs")
    parser.add_argument("--pretrained",action='store_true', help="whether use pretrained weights")
    parser.add_argument("--model_path", type=str, default="ckpt/resnet34/resnet_epoch_1.pth", help="path to load pretrained weights")
    parser.add_argument("--start_epoch", type=int, default=1, help="Start Training Epoch")
    params = parser.parse_args()
    print(params) 
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", 
            "other_vehicle", "animal", "emergency_vehicle"]
    batch_size = params.batch_size
    bev_shape = (672, 672, 3)
    epochs = params.epochs
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
    class_weights = class_weights.to(device)

    train_data_folder = os.path.join(params.data_folder, "bev_train_data")
    input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
    target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))
    map_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_map.png")))
    train_dataset = BEVImageDataset(input_filepaths, target_filepaths, map_filepaths)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    
    # validation_data_folder = os.path.join(params.data_folder, "./bev_validation_data")
    # val_input_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_input.png")))
    # val_target_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target.png")))
    # #map_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_map.png")))
    # val_dataset = BEVImageDataset(val_input_filepaths, val_target_filepaths)
    # valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=os.cpu_count())

    #model = get_unet_model(in_channels=3, num_output_classes=len(classes)+1)
    model = DepthCompletionNet(in_channels=6, num_out_channels=len(classes)+1, res_layers=params.reslayers)
    if params.pretrained:
        state = torch.load(params.model_path)
        model.load_state_dict(state)
    if params.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.7) #optimizer, gamma, last_epoch
    train_losses = []
    ckpt_path = os.path.join(params.ckpt, "resnet{}/".format(params.reslayers))
    os.makedirs(ckpt_path, exist_ok=True)

    for epoch in range(params.start_epoch, epochs+1):
        ## Train
        print("Epoch", epoch)
        train_epoch_losses = []
        model.train()
        scheduler.step()
        for ii, (X, target, sample_ids) in enumerate(tqdm(trainloader)):
            X = X.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)
            optim.zero_grad()
            loss.backward()
            optim.step()        
            train_epoch_losses.append(loss.detach().cpu().numpy())
            #if ii == 0:            
            #   visualize_predictions(X, prediction, target)    
        train_loss = np.mean(train_epoch_losses)
        train_losses.append(train_loss)
        #print("Train Loss:", train_loss)
        checkpoint_filename = "resnet_epoch_{}.pth".format(epoch)
        checkpoint_filepath = os.path.join(ckpt_path, checkpoint_filename)
        if params.multi_gpu:
            torch.save(model.module.state_dict(), checkpoint_filepath)
        else:
            torch.save(model.state_dict(), checkpoint_filepath)
        print("save model in {}".format(checkpoint_filepath))
    

