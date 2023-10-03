#!/usr/bin/env python3
"""
Author: gabriel, ilyi
Date: 2022
"""
import os
import time
import argparse
import torch
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from tqdm import tqdm
from PIL import Image
from PIL.ImageFile import ImageFile
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from data import WasteDataset
from utils import (
    display, 
    plot_validation_loss, 
    plot_validation_accuracy, 
    plot_training_loss, 
    plot_training_accuracy, 
    plot_time_per_epoch, 
    accuracy, 
    increment_path,
    load_model
)

CUDA_LAUNCH_BLOCKING = 1



def build_dataloaders(data_path, classes: List, transform, batch_size, train_ratio=0.8):
    print("Creating dataset ...")
    dataset = WasteDataset(data_path, classes=classes, transform=transform)
     
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    print("Loading dataloaders ...")
    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                   'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)}

    return dataloaders

def train(trial, args, params, dataloaders, classes, model, optimizer, exp_lr_scheduler):
    print("Training started...")
    time_per_epoch = []
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    valid_loss_min = np.Inf

    train_dataloader, val_dataloader = dataloaders.values()

    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()

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
            data, target = data.to(args.device), target.to(args.device)
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

        # Optuna
        trial.report(valid_loss_min, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    if args.plot:
        plot_validation_loss(val_loss, os.path.join(incremented_path,"val_loss.png"))
        plot_validation_accuracy(val_acc, os.path.join(incremented_path,"val_acc.png"))
        plot_training_loss(train_loss, os.path.join(incremented_path,"train_loss.png"))
        plot_training_accuracy(train_acc, os.path.join(incremented_path,"train_acc.png"))
        plot_time_per_epoch(time_per_epoch, os.path.join(incremented_path,"time_per_epoch.png"))

    return valid_loss_min

def objective(trial, args, classes):
    # Hyperparameters we want optimize    
    params = {
        "lr": trial.suggest_loguniform('lr', 1e-4, 1e-2),
        "optimizer_name": trial.suggest_categorical('optimizer_name', ["SGD", "Adam", "Adagrad", "RMSprop"]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }
    if args.model_name == "SSWQ":
        params["dropout"] = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5]),
        params["hidden_layer1"] = trial.suggest_categorical("hidden_layer1", [512, 1024, 2048]),
        params["hidden_layer2"] = trial.suggest_categorical("hidden_layer2", [512, 1024, 2048])

        model = SSWQ(num_classes, dropout=params["dropout"], hidden_layer1=params["hidden_layer1"], hidden_layer2=params["hidden_layer2"])
        model = model.to(device)
    else:
        model = load_model(args.model_name, len(classes))

    # Define criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, params["optimizer_name"])(model.parameters(), lr=params["lr"])

    if params["optimizer_name"] == "SGD":
        params["momentum"] = trial.suggest_loguniform('momentum', 0.85, 0.95)
        # Configure optimizer again for SGD
        optimizer = getattr(torch.optim, params["optimizer_name"])(model.parameters(), lr=params["lr"],
                                                                   momentum=params["momentum"])

    # Load data
    transform = transforms.Compose(
        [
            transforms.Resize((226, 226)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataloaders = build_dataloaders(args.data_path, classes, transform, params["batch_size"], train_ratio=0.8)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train a model
    best_loss = train(trial, args, params, dataloaders, classes, model, optimizer, exp_lr_scheduler)  
    return best_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument("--step_size", type=int, default=7, help="step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
    parser.add_argument("--data_path", "--datapath", type=str, help="path to dataset")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--plot", type=bool, default=False, help="plot graphs")
    parser.add_argument("--model_name", type=str, default="efficientnet_b3", help="model name")

    parser.add_argument_group("Optuna")
    parser.add_argument("--num_trials", type=int, default=10, help="number of trials")
    parser.add_argument("--n_startup_trials", type=int, default=2, help="number of startup trials")
    parser.add_argument("--n_warmup_steps", type=int, default=5, help="number of warmup steps")
    parser.add_argument("--interval_steps", type=int, default=3, help="interval steps")

    opt = parser.parse_args()

    classes = ["glas", "organic", "paper", "restmuell", "wertstoff"]


    # OPTUNA
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=opt.n_startup_trials, 
            n_warmup_steps=opt.n_warmup_steps, 
            interval_steps=opt.interval_steps
        ),
        direction='minimize')
    study.optimize(lambda trial: objective(trial, opt, classes), n_trials=opt.num_trials)

    trial = study.best_trial
    print()

    summary_filename = os.path.join(output_dir, "hpo_summary.txt")
    with open(summary_filename, "w") as summary_file:
            summary_file.write(f"--- {study.study_name} ---\n")
            summary_file.write(f'Number of finished trials: {len(study.trials)}\n')
            summary_file.write(f'Best trial: {trial.number}\n')
            summary_file.write(f'Loss: {trial.value}\nBest hyperparameters: {trial.params}')

    print(f"Hyperparameter optimization finished. Results saved in {summary_filename}.")