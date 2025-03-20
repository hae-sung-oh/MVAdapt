import importlib
import math
import os
import random
import re
import trace
import traceback
import numpy as np

from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import wandb
from mvadapt_v3 import MVAdapt
from mvadapt_data import MVAdaptDataset

def compute_wp_accuracy(predicted, gt, tolerance=0.05):
    dimension_accuracy = torch.abs(predicted - gt) < tolerance 

    correct = dimension_accuracy.all(dim=1)
    return correct.sum().item() / gt.size(0)

def compute_mae(predicted, gt):
    return torch.mean(torch.abs(predicted - gt)).item()

def lossfn(pred, gt):
    return torch.mean(torch.abs(pred - gt))

def train(model, args, dataset):
    print("Training Model")
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # type: ignore
    loss_fn = lossfn

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())
    data_loader = DataLoader(dataset, 
                         batch_size=args.batch_size, 
                         shuffle=True, 
                        #  num_workers=os.cpu_count() - 8,  # type: ignore
                         pin_memory=True, 
                         pin_memory_device=args.device,
                        #  generator=g_cuda, 
                        )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Wrap the DataLoader in a progress bar
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")

        for data in progress_bar:
            gt_wp = data['gt_waypoint'].to(args.device, dtype=torch.float32, non_blocking=True)
            x = data['scene_features'].to(args.device, dtype=torch.float32, non_blocking=True)
            phys = data['physics_params'].to(args.device, dtype=torch.float32, non_blocking=True)
            gear = data['gear_params'].to(args.device, dtype=torch.float32, non_blocking=True)
            target = data['target_point'].to(args.device, dtype=torch.float32, non_blocking=True)
            
            optimizer.zero_grad()
            predicted = model.forward(x, target, phys, gear)
            loss = loss_fn(predicted, gt_wp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_accuracy = compute_wp_accuracy(predicted, gt_wp, tolerance=0.1)
            total_correct += batch_accuracy * gt_wp.size(0)
            total_samples += gt_wp.size(0)

            # Update progress bar with loss information
            progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_correct / total_samples
        # wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": avg_accuracy})
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

def validate(model, args, dataset) -> None:
    print("Validating Model")
    model.eval()
    loss_fn = lossfn
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=g_cuda, num_workers=os.cpu_count()-8) # type: ignore
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             pin_memory=True,
                             pin_memory_device=args.device,
                            #  num_workers=os.cpu_count()-8 # type: ignore
                             shuffle=True, )
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Validation"):
            gt_wp = data['gt_waypoint'].to(args.device, dtype=torch.float32)
            x = data['scene_features'].to(args.device, dtype=torch.float32)
            phys = data['physics_params'].to(args.device, dtype=torch.float32)
            gear = data['gear_params'].to(args.device, dtype=torch.float32)
            target = data['target_point'].to(args.device, dtype=torch.float32)
            
            predicted = model.inference(x, target, phys, gear)
            loss = loss_fn(predicted, gt_wp)
            total_loss += loss.item()
            
            batch_accuracy = compute_wp_accuracy(predicted, gt_wp, tolerance=0.05)
            total_correct += batch_accuracy * gt_wp.size(0)
            total_samples += gt_wp.size(0)
            
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_samples
    print(f"Validation Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    wandb.log({"validation_loss": avg_loss, "validation_accuracy": avg_accuracy})

def main():
    parser = ArgumentParser()
    parser.add_argument("--vehicle_indices", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset")
    parser.add_argument("--base_model", type=str, default="/path/to/base_model")
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--gear_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--process_batch", type=int, default=32)
    parser.add_argument("--save_data", type=str, default="preprocessed_data.pt")
    parser.add_argument("--save_model", type=str, default="mvadapt_model.pth")
    parser.add_argument("--load_data", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--remove_crashed", type=bool, default=False)
    parser.add_argument("--remove_imperfect", type=bool, default=False)
    parser.add_argument("--move_dup_dir", type=str, default=None)

    args = parser.parse_args()
    
    train_set = MVAdaptDataset(args.root_dir)
    test_set = MVAdaptDataset(args.root_dir)
    
    loaded = False
    if args.load_data is not None and args.load_data != "None":
        try:
            train_set.load(f"{args.load_data}_train.pkl")
            test_set.load(f"{args.load_data}_test.pkl")
            loaded = True
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            loaded = False
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            loaded = False
            
    if not loaded:
        train_set.initialize(args, 'train', args.remove_crashed, args.remove_imperfect, args.move_dup_dir)
        test_set.initialize(args, 'val', args.remove_crashed, args.remove_imperfect, args.move_dup_dir)
        
    if args.save_data is not None and args.save_data != "None":
        train_set.save(args.save_data + "_train.pkl")
        test_set.save(args.save_data + "_test.pkl")
        
    # wandb.init(project="MVAdapt-Training", config=args.__dict__)
    loaded = False
    model = importlib.import_module(f'team_code_mvadapt.mvadapt_{args.version}').MVAdapt(train_set.config, args).to(args.device)
    if args.load_model is not None and args.load_model != "None":
        try:
            print("Loading Model")
            model.load(args.load_model)
            loaded = True
        except FileNotFoundError:
            print(f"File {args.load_model} not found")
            loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            loaded = False
            
    if not loaded:
        print("Train Model")
        train(model, args, train_set)
        
    if args.save_model is not None and args.save_model != "None":
        model.save(args.save_model)
        print(f"Model saved to {args.save_model}")
        
    validate(model, args, test_set)
    
    if args.debug:
        print("Exporting debug data")
        model.debug(os.getenv("WORK_DIR", '.') + f'/debug_{args.version}', args.base_model, train_set, 200)
    
    # wandb.finish()

if __name__ == "__main__":
    main()
