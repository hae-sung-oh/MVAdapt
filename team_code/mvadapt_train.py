import math
import os
import random
import re
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
import carla

from data import CARLA_Data
from vehicle_config import VehicleConfig
from config import GlobalConfig
from basemodel_agent import BasemodelAgent
from mvadapt import MVAdapt

def normalize(value, max, min):
    if isinstance(value, (list, tuple)):
        return type(value)(normalize(v, max, min) for v in value)
    else:
        return (value - min) / (max - min)

def calculate_wheelbase(vehicle):
    front_position_x = vehicle["physics"]["wheels"][0]["position"][0]
    rear_position_x = vehicle["physics"]["wheels"][2]["position"][0]
    return abs(front_position_x - rear_position_x)

def pad_data(data, max_len):
    result = [0] * max_len  # Initialize with zeros
    data = list(np.ravel(data))
    result[:len(data)] = [item for item in data]
    return result

def to_device(device, *kwargs):
    return tuple(t.to(device) for t in kwargs)

def load_physics_data(vehicle_index, device='cuda:0', norm=True):
    physics_prop, gear_prop = [], []
    v_config = VehicleConfig()
    g_config = GlobalConfig()
    item = v_config.config_list[vehicle_index]

    wheelbase = calculate_wheelbase(item)
    if norm:
        item["vehicle_extent"] = normalize(item["vehicle_extent"], g_config.max_extent, g_config.min_extent)
        item["physics"]["torque_curve"] = normalize(pad_data(item["physics"]["torque_curve"], 2*g_config.max_torque_num), g_config.max_torque_curve, g_config.min_torque_curve)
        item["physics"]["max_rpm"] = normalize(item["physics"]["max_rpm"], g_config.max_max_rpm, g_config.min_max_rpm)
        item["physics"]["wheels"][0]["radius"] = normalize(item["physics"]["wheels"][0]["radius"], g_config.max_radius, g_config.min_radius)
        item["physics"]["mass"] = normalize(item["physics"]["mass"], g_config.max_mass, g_config.min_mass)
        wheelbase = normalize(wheelbase, g_config.max_wheelbase, g_config.min_wheelbase)

    physics_prop.extend(item["vehicle_extent"])
    physics_prop.extend(item["physics"]["torque_curve"])
    physics_prop.append(item["physics"]["max_rpm"])
    physics_prop.append(item["physics"]["wheels"][0]["radius"])
    physics_prop.extend(list(item["physics"]["center_of_mass"]))
    physics_prop.append(item["physics"]["mass"])
    physics_prop.append(wheelbase)

    for gear in item["physics"]["forward_gears"]:
        gear_prop.extend([gear["ratio"], gear["up_ratio"], gear["down_ratio"]])
    gear_prop = pad_data(gear_prop, 3 * g_config.max_gear_num)

    return torch.tensor(physics_prop, dtype=torch.float32).to(device), \
           torch.tensor(gear_prop, dtype=torch.float32).to(device)

def process_data(data, agent, device) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb = data['rgb'].to(device, dtype=torch.float32)
    lidar = data['lidar'].to(device, dtype=torch.float32)
    target = data['target_point'].to(device, dtype=torch.float32)
    ego_vel = data['speed'].to(device, dtype=torch.float32).unsqueeze(1)
    command = data['command'].to(device, dtype=torch.float32)

    gt_wp = data['ego_waypoints'].to(device, dtype=torch.float32).flatten()

    with torch.no_grad():
        pred_wp = agent.nets[0].forward(
            rgb=rgb, lidar_bev=lidar, target_point=target, ego_vel=ego_vel, command=command
        )[0]
        # pred_control = agent.nets[0].control_pid(pred_wp, data['speed'])

    # gt_control = [
    #     data['control']['throttle'].item(),
    #     data['control']['steer'].item(),
    #     data['control']['brake'].item()
    # ]
    # pred_control_tensor = [pred_control[1], pred_control[0], float(pred_control[2])]
    return gt_wp, pred_wp

def stack_data(*kwargs) :
    return tuple(torch.stack(tensors) for tensors in kwargs)

def save_tensors(file_path, *tensors):
    torch.save(tensors, file_path)
    print(f"Saved tensors to {file_path}")

def get_data(args):
    agent = BasemodelAgent(args.base_model, verbose=args.verbose)
    config = agent.config

    gt_wp_train, pred_wp_train, physics_train, gear_train = [], [], [], []
    gt_wp_val, pred_wp_val, physics_val, gear_val = [], [], [], []

    for vehicle_index in eval(args.vehicle_indices):
        config.update_vehicle(vehicle_index)
        config.initialize(root_dir=args.root_dir, vehicle_index=vehicle_index, verbose=args.verbose)

        train_set = CARLA_Data(config.train_data, config, verbose=args.verbose)
        val_set = CARLA_Data(config.val_data, config, verbose=args.verbose)

        dataloader_train = DataLoader(train_set, batch_size=1, shuffle=True)
        dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False)

        physics_prop, gear_prop = load_physics_data(vehicle_index, args.device)

        # Training Data
        for data in tqdm(dataloader_train, desc=f"Processing Train Data - Vehicle {vehicle_index}"):
            gt_wp, pred_wp = process_data(data, agent, args.device)
            gt_wp_train.append(gt_wp)
            pred_wp_train.append(pred_wp)
            physics_train.append(physics_prop)
            gear_train.append(gear_prop)

        # Validation Data
        for data in tqdm(dataloader_val, desc=f"Processing Val Data - Vehicle {vehicle_index}"):
            gt_wp, pred_wp = process_data(data, agent, args.device)
            gt_wp_val.append(gt_wp)
            pred_wp_val.append(pred_wp)
            physics_val.append(physics_prop)
            gear_val.append(gear_prop)

    data_ = stack_data(gt_wp_train, pred_wp_train, physics_train, gear_train, 
                    gt_wp_val, pred_wp_val, physics_val, gear_val)
    
    if args.save_data is not None and args.save_data != "None":
        save_tensors(args.save_data, data_)
        if args.verbose:
            print(f"Saved preprocessed data to {args.save_data}")

    return data_

def Loss_fn(pred, gt):
    if isinstance(pred, carla.VehicleControl):
        pred = torch.tensor([pred.throttle, pred.steer, pred.brake], dtype=torch.float32, requires_grad=True)
    return F.mse_loss(pred, gt)

def compute_control_accuracy(predicted, gt, tolerance=0.05):
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

def train(args, gt_controls, pred_controls, physics, gear) -> MVAdapt:
    print("Training Model")
    model = MVAdapt(args.dim0, args.dim1, args.dim2, args.dim3, args.physics_dim, args.max_gear_num, args.gear_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # type: ignore
    loss_fn = Loss_fn

    dataset = TensorDataset(gt_controls, pred_controls, physics, gear)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

        for gt, pred, phys, g in progress_bar:
            optimizer.zero_grad()
            predicted = model(pred, phys, g)
            loss = loss_fn(predicted, gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_accuracy = compute_control_accuracy(predicted, gt, tolerance=0.1)
            total_correct += batch_accuracy * gt.size(0)
            total_samples += gt.size(0)

            # Update progress bar with loss information
            progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_correct / total_samples
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": avg_accuracy})
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Simulated annealing step
        if temperature > min_temperature:
            with torch.no_grad():
                for param in model.parameters():
                    # Save the current weights
                    original_weights = param.clone()

                    # Perturb weights with random noise
                    noise = torch.randn_like(param) * (temperature / initial_temperature)
                    param.add_(noise)

                    # Compute the new loss
                    perturbed_loss = 0
                    for gt, pred, phys, g in data_loader:
                        predicted = model(pred, phys, g)
                        perturbed_loss += loss_fn(predicted, gt).item()

                    # Acceptance probability
                    delta_loss = perturbed_loss - total_loss
                    acceptance_prob = torch.exp(torch.tensor(-delta_loss / temperature)) if delta_loss > 0 else torch.tensor(1.0)

                    # Decide whether to accept or reject the perturbed weights
                    if torch.rand(1).item() > acceptance_prob:
                        param.copy_(original_weights)  # Reject the perturbation

            # Cool down the temperature
            temperature *= cooling_rate
            print(f"Simulated Annealing: Temperature decreased to {temperature:.4f}")

    if args.save_model is not None and args.save_model != "None":
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    return model


def validate(model, args, gt_controls_val, pred_controls_val, physics_val, gear_val) -> None:
    print("Validating Model")
    model.eval()
    loss_fn = Loss_fn
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    dataset = TensorDataset(gt_controls_val, pred_controls_val, physics_val, gear_val)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    with torch.no_grad():
        for gt, pred, phys, g in tqdm(data_loader, desc="Validation"):
            predicted = model(pred, phys, g)
            loss = loss_fn(predicted, gt)
            total_loss += loss.item()
            
            batch_accuracy = compute_control_accuracy(predicted, gt, tolerance=0.05)
            total_correct += batch_accuracy * gt.size(0)
            total_samples += gt.size(0)
            
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_samples
    print(f"Validation Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    wandb.log({"validation_loss": avg_loss, "validation_accuracy": avg_accuracy})

def main():
    parser = ArgumentParser()
    parser.add_argument("--vehicle_indices", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset")
    parser.add_argument("--base_model", type=str, default="/path/to/base_model")
    parser.add_argument("--dim0", type=int, default=32)
    parser.add_argument("--dim1", type=int, default=32)
    parser.add_argument("--dim2", type=int, default=64)
    parser.add_argument("--dim3", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--physics_dim", type=int, default=18)
    parser.add_argument("--max_gear_num", type=int, default=8)
    parser.add_argument("--gear_dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_data", type=str, default="preprocessed_data.pt")
    parser.add_argument("--save_model", type=str, default="mvadapt_model.pth")
    parser.add_argument("--load_data", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)

    args = parser.parse_args()
    
    loaded = False
    if args.load_data is not None and args.load_data != "None":
        try:
            gt_wp_train, pred_wp_train, physics_train, gear_train, \
            gt_wp_val, pred_wp_val, physics_val, gear_val = to_device(args.device, *torch.load(args.load_data, weights_only=True))
            loaded = True
        except FileNotFoundError:
            print(f"File {args.load_data} not found")
            loaded = False
        except Exception as e:
            print(f"Error loading data: {e}")
            loaded = False
            
    if not loaded:
        gt_wp_train, pred_wp_train, physics_train, gear_train, gt_wp_val, pred_wp_val, physics_val, gear_val = get_data(args)

    wandb.init(project="MVAdapt-Training", config=args.__dict__)
    loaded = False
    if args.load_model is not None and args.load_model != "None":
        try:
            print("Loading Model")
            model = MVAdapt(args.dim0, args.dim1, args.dim2, args.dim3, args.physics_dim, args.max_gear_num, args.gear_dim)
            model.load(args.load_model)
            loaded = True
        except FileNotFoundError:
            print(f"File {args.load_model} not found")
            loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            loaded = False
            
    if not loaded:
        model = train(args, gt_wp_train, pred_wp_train, physics_train, gear_train)
        
    validate(model, args, gt_wp_val, pred_wp_val, physics_val, gear_val)
    wandb.finish()

if __name__ == "__main__":
    main()
