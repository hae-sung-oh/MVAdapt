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

# def train(args, gt_controls, pred_controls, physics, gear) -> MVAdapt:
#     print("Training Model")
#     model = MVAdapt(args.dim0, args.dim1, args.dim2, args.dim3, args.physics_dim, args.max_gear_num, args.gear_dim)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr) # type: ignore
#     loss_fn = Loss_fn
    
#     dataset = TensorDataset(gt_controls, pred_controls, physics, gear)
#     data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
#     for epoch in range(args.epochs):
#         model.train()
#         total_loss = 0
#         total_correct = 0
#         total_samples = 0
        
#         # Wrap the DataLoader in a progress bar
#         progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")
        
#         for gt, pred, phys, g in progress_bar:
#             optimizer.zero_grad()
#             predicted = model(pred, phys, g)
#             loss = loss_fn(predicted, gt)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#             batch_accuracy = compute_control_accuracy(predicted, gt, tolerance=0.1)
#             total_correct += batch_accuracy * gt.size(0)
#             total_samples += gt.size(0)
            
#             # Update progress bar with loss information
#             progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)

#         avg_loss = total_loss / len(data_loader)
#         avg_accuracy = total_correct / total_samples
#         wandb.log({"epoch": epoch + 1, "train_loss": avg_loss,  "train_accuracy": avg_accuracy})
#         print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

#     if args.save_model is not None and args.save_model != "None":
#         torch.save(model.state_dict(), args.save_model)
#         print(f"Model saved to {args.save_model}")
        
#     return model

def lossfn(pred, gt):
    return torch.mean(torch.abs(pred - gt))

def train(model, args, dataset) -> MVAdapt:
    print("Training Model")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # type: ignore
    loss_fn = lossfn

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())
    data_loader = DataLoader(dataset, 
                         batch_size=args.batch_size, 
                         shuffle=True, 
                         num_workers=4, 
                         pin_memory=True, 
                         generator=g_cuda, 
                         prefetch_factor=2)

    # Simulated annealing parameters
    initial_temperature = 100
    cooling_rate = 0.9
    temperature = initial_temperature
    min_temperature = 1e-3

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Wrap the DataLoader in a progress bar
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")

        for data in progress_bar:
            gt_wp = data['gt_waypoint'].to(args.device, dtype=torch.float32)
            x = data['scene_features'].to(args.device, dtype=torch.float32)
            phys = data['physics_params'].to(args.device, dtype=torch.float32)
            gear = data['gear_params'].to(args.device, dtype=torch.float32)
            target = data['target_point'].to(args.device, dtype=torch.float32)
            
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
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": avg_accuracy})
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # # Simulated annealing step
        # if temperature > min_temperature:
        #     with torch.no_grad():
        #         for param in model.parameters():
        #             # Save the current weights
        #             original_weights = param.clone()

        #             # Perturb weights with random noise
        #             noise = torch.randn_like(param) * (temperature / initial_temperature)
        #             param.add_(noise)

        #             # Compute the new loss
        #             perturbed_loss = 0
        #             for gt, pred, phys, g in data_loader:
        #                 predicted = model(pred, phys, g)
        #                 perturbed_loss += loss_fn(predicted, gt).item()

        #             # Acceptance probability
        #             delta_loss = perturbed_loss - total_loss
        #             acceptance_prob = torch.exp(torch.tensor(-delta_loss / temperature)) if delta_loss > 0 else torch.tensor(1.0)

        #             # Decide whether to accept or reject the perturbed weights
        #             if torch.rand(1).item() > acceptance_prob:
        #                 param.copy_(original_weights)  # Reject the perturbation

        #     # Cool down the temperature
        #     temperature *= cooling_rate
        #     print(f"Simulated Annealing: Temperature decreased to {temperature:.4f}")

    return model


def validate(model, args, dataset) -> None:
    print("Validating Model")
    model.eval()
    loss_fn = lossfn
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=g_cuda, num_workers=4)
    
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

    args = parser.parse_args()
    
    master_set = MVAdaptDataset(args.root_dir)
    train_set = MVAdaptDataset(args.root_dir)
    test_set = MVAdaptDataset(args.root_dir)
    
    loaded = False
    if args.load_data is not None and args.load_data != "None":
        try:
            train_set.load(f"{args.load_data}_train.pkl")
            test_set.load(f"{args.load_data}_test.pkl")
            # train_set.load(args.load_data)
            # test_set.load(args.load_data)
            loaded = True
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            loaded = False
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            loaded = False
            
    if not loaded:
        train_set.initialize(args, 'train')
        test_set.initialize(args, 'val')
        ###################################################################################################
        train_set.save('/home/ohs-dyros/gitRepo/MVAdapt/dataset/train_set.pkl')
        test_set.save('/home/ohs-dyros/gitRepo/MVAdapt/dataset/test_set.pkl')
        train_set = train_set.sample_data_per_vehicle(50000, exclude=[1, 2, 3, 5, 7, 8, 9, 11, 12, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35])
        test_set = test_set.sample_data_per_vehicle(50000, exclude=[1, 2, 3, 5, 7, 8, 9, 11, 12, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35])
        ###################################################################################################
        
    if args.save_data is not None and args.save_data != "None":
        train_set.save(args.save_data + "_train.pkl")
        test_set.save(args.save_data + "_test.pkl")
        
    wandb.init(project="MVAdapt-Training", config=args.__dict__)
    loaded = False
    model = importlib.import_module(f'team_code_mvadapt.mvadapt_{args.version}').MVAdapt(train_set.config, args)
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
        train(model, args, train_set)
        
    if args.save_model is not None and args.save_model != "None":
        model.save(args.save_model)
        print(f"Model saved to {args.save_model}")
        
    validate(model, args, test_set)
    
    if args.debug:
        print("Exporting debug data")
        model.debug(os.environ("WORK_DIR", '.') + f'/debug_{args.version}', args.base_model, train_set, 200)
    
    wandb.finish()

if __name__ == "__main__":
    main()
