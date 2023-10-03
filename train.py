#!/usr/bin/env python3
"""
Author: gabriel, ilyi
Date: 2022
"""

import os
import time
from typing import List
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageFile import ImageFile
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

from data import WasteDataset
from utils import display, plot_validation_loss, plot_validation_accuracy, plot_training_loss, plot_training_accuracy, plot_time_per_epoch, accuracy, increment_path, load_model

# ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(args, dataloaders, classes):
    print("Training started...")
    time_per_epoch = []
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    valid_loss_min = np.Inf

    train_dataloader, val_dataloader = dataloaders.values()

    print("Loading model ...")
    model = load_model(model_name=args.model_name, num_classes=len(classes))
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(1, args.n_epochs + 1):
        #----- TRAINING -----#
        start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (data, target) in pbar_train:
            pbar_train.set_description(f"[Train] Epoch {epoch}/{args.n_epochs} | Batch {batch_idx+1}/{len(train_dataloader)}")

            target = target.type(torch.LongTensor)
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target).item()
            total += target.size(0)

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / len(train_dataloader))
        print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}")

        #----- VALIDATION -----#
        batch_loss = 0
        correct = 0
        total = 0
        pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in pbar_val:
                pbar_val.set_description(f"[Val] Epoch {epoch}/{args.n_epochs} | Batch {batch_idx+1}/{len(pbar_val)}")
                data, target = data.to(args.device), target.to(args.device)
                outputs = model(data)
                loss = criterion(outputs, target)
                batch_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred == target).item()
                total += target.size(0)
            val_acc.append(100 * correct / total)
            val_loss.append(batch_loss / len(val_dataloader))
            improvement = batch_loss < valid_loss_min            
            print(f"validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct / total):.4f}\n")
            
            # Saving the model
            if improvement:
                valid_loss_min = batch_loss
                incremented_path = increment_path(os.path.join(args.output_dir, "train"))
                os.makedirs(incremented_path)
                torch.save(model.state_dict(), os.path.join(incremented_path,"best.pt"))

        model.train()
        exp_lr_scheduler.step()
        end = time.time()
        time_per_epoch.append(round((end - start), 2))

    if args.plot:
        plot_validation_loss(val_loss, os.path.join(incremented_path,"val_loss.png"))
        plot_validation_accuracy(val_acc, os.path.join(incremented_path,"val_acc.png"))
        plot_training_loss(train_loss, os.path.join(incremented_path,"train_loss.png"))
        plot_training_accuracy(train_acc, os.path.join(incremented_path,"train_acc.png"))
        plot_time_per_epoch(time_per_epoch, os.path.join(incremented_path,"time_per_epoch.png"))

    print("Training Done.")



def build_dataloaders(data_path, classes:List, transform, batch_size, train_ratio=0.8):
    print("Creating dataset ...")
    dataset = WasteDataset(data_path, classes=classes, transform=transform)
     
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    print("Loading dataloaders ...")
    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                   'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)}


    return dataloaders
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument("--lr", type=float, default=0.001974473, help="learning rate")
    parser.add_argument("--step_size", type=int, default=7, help="step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
    parser.add_argument("--data_path", "--datapath", type=str, help="path to dataset")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--plot", type=bool, default=False, help="plot graphs")
    parser.add_argument("--model_name", type=str, default="resnet18", help="model name")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    parser.add_argument("--image_size", "--imgsz", type=int, default=224, help="image size")
    opt = parser.parse_args()



    classes = ["glas", "organic", "paper", "restmuell", "wertstoff"]


    # Custom Transform for dataset
    transform = transforms.Compose(
        [
            transforms.Resize((opt.image_size+2, opt.image_size+2)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataloaders = build_dataloaders(data_path = opt.data_path,
                                    classes = classes, 
                                    transform = transform, 
                                    batch_size = opt.batch_size, 
                                    train_ratio= opt.train_ratio)   


    train(opt, dataloaders=dataloaders, classes=classes)
