import os
import pickle
import shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
from config import GlobalConfig
from data import CARLA_Data
from vehicle_config import VehicleConfig
from basemodel_agent import BasemodelAgent
import random
from collections import defaultdict

all_vehicles = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35]


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

def unroot(path_list, root_dir):
    for i in range(len(path_list)):
        path_list[i] = os.path.relpath(path_list[i], root_dir)
    return path_list

def process_data(data, agent, device, root_dir):
    batch_size = data['rgb'].size(0)
    
    gt_con = []
    bs_con = []
    
    rgb = data['rgb'].to(device, dtype=torch.float32)
    lidar = data['lidar'].to(device, dtype=torch.float32)
    rgb_path = data['rgb_path']
    lidar_path = data['lidar_path']
    target = data['target_point'].to(device, dtype=torch.float32)
    ego_vel = data['speed'].to(device, dtype=torch.float32)
    command = data['command'].to(device, dtype=torch.float32)
    temp_gt_con = data['control']

    gt_wp = data['ego_waypoints'].to(device, dtype=torch.float32).reshape(batch_size, -1)

    with torch.no_grad():
        result = agent.nets[0].forward(
            rgb=rgb, 
            lidar_bev=lidar, 
            target_point=target, 
            ego_vel=ego_vel.unsqueeze(1), 
            command=command
        )
        bs_wp = result[0]
        
        s_feature = result[10].reshape(batch_size, -1)
        

    for i in range(batch_size):
        gt_con.append(
            np.array([temp_gt_con['throttle'][i].item(), temp_gt_con['steer'][i].item(), temp_gt_con['brake'][i].item()], dtype=float))
        
        temp_bs_con = agent.nets[0].control_pid(bs_wp[i].unsqueeze(0), ego_vel[i].unsqueeze(0))
        bs_con.append(np.array([temp_bs_con[1], temp_bs_con[0], float(temp_bs_con[2])], dtype=float))
    
    gt_wp = gt_wp.cpu().numpy()
    bs_wp = bs_wp.reshape(batch_size, -1).cpu().numpy()
    s_feature = s_feature.cpu().numpy()
    lidar = lidar.cpu().numpy()
    target = target.cpu().numpy()
    ego_vel = ego_vel.cpu().numpy()
    command = command.cpu().numpy()
    
    return list(gt_wp), list(bs_wp), gt_con, bs_con, list(s_feature), unroot(rgb_path, root_dir), unroot(lidar_path, root_dir), list(target), list(ego_vel), list(command)

def move_duplicated_data(root_dir, backup_dir):
    towns = os.listdir(root_dir)
    for town in towns:
        if os.path.isfile(os.path.join(root_dir, town)):
            continue 

        v_indices = os.listdir(os.path.join(root_dir, town))
        for index in v_indices:
            folders = os.listdir(os.path.join(root_dir, town, index))
            temp = []

            for folder in folders:
                if os.path.isfile(os.path.join(root_dir, town, index, folder)):
                    continue
                temp.append(os.path.join(root_dir, town, index, folder))

            temp.sort()

            for i in range(len(temp) - 1):
                si = '_'.join(temp[i].split('_')[:-5])
                si_ = '_'.join(temp[i+1].split('_')[:-5])

                if si == si_:
                    lidar_path_i = os.path.join(temp[i], 'lidar')
                    lidar_path_i_ = os.path.join(temp[i+1], 'lidar')

                    num_i = len(os.listdir(lidar_path_i)) if os.path.exists(lidar_path_i) else 0
                    num_i_ = len(os.listdir(lidar_path_i_)) if os.path.exists(lidar_path_i_) else 0

                    remove = temp[i] if num_i < num_i_ else temp[i+1]

                    dir_name = os.path.join(backup_dir, town, index, os.path.basename(remove))
                    os.makedirs(dir_name, exist_ok=True)  

                    if os.path.exists(remove): 
                        for file in os.listdir(remove):  
                            shutil.move(os.path.join(remove, file), dir_name)
                            
def remove_crashed_data(root_dir):
    config = GlobalConfig()
    config.initialize(root_dir=root_dir)
    temp_dataset = CARLA_Data(config.train_data, config, clear_crashed=True, clear_imperfect=True)
    del config, temp_dataset

class MVAdaptDataset(Dataset):
    def __init__(self, root_dir, config=None, vehicle_config=None):
        self.config = GlobalConfig() if config is None else config
        self.vehicle_config = VehicleConfig() if vehicle_config is None else vehicle_config
        self.root_dir = root_dir

        # Data for MVAdapt model
        self.vehicle_indices = []
        self.gt_waypoints = []
        self.bs_waypoints = []
        self.gt_controls = []
        self.bs_controls = []
        self.scene_features = []
        self.physics_params = []
        self.gear_params = []

        # Data for baseline model
        self.rgb_paths = []
        self.lidar_paths = []
        self.target_points = []
        self.ego_vels = []
        self.commands = []

    def __len__(self):
        return len(self.vehicle_indices)

    def __getitem__(self, idx):
        data = {}
        data['vehicle_index'] = self.vehicle_indices[idx]
        data['gt_waypoint'] = self.gt_waypoints[idx]
        data['bs_waypoint'] = self.bs_waypoints[idx]
        data['gt_control'] = self.gt_controls[idx]
        data['bs_control'] = self.bs_controls[idx]
        data['scene_features'] = self.scene_features[idx]
        data['physics_params'] = self.physics_params[idx]
        data['gear_params'] = self.gear_params[idx]
        data['rgb'] = np.load(os.path.join(self.root_dir, self.rgb_paths[idx]))
        data['lidar'] = np.load(os.path.join(self.root_dir, self.lidar_paths[idx]))
        data['target_point'] = self.target_points[idx]
        data['ego_vel'] = self.ego_vels[idx]
        data['command'] = self.commands[idx]
        data['rgb_path'] = self.rgb_paths[idx]
        data['lidar_path'] = self.lidar_paths[idx]

        return data

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved MVAdapt Dataset to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        self.config = dataset.config
        self.vehicle_config = dataset.vehicle_config

        self.vehicle_indices = dataset.vehicle_indices
        self.gt_waypoints = dataset.gt_waypoints
        self.bs_waypoints = dataset.bs_waypoints
        self.gt_controls = dataset.gt_controls
        self.bs_controls = dataset.bs_controls
        self.scene_features = dataset.scene_features
        self.physics_params = dataset.physics_params
        self.gear_params = dataset.gear_params

        self.rgb_paths = dataset.rgb_paths
        self.lidar_paths = dataset.lidar_paths
        self.target_points = dataset.target_points
        self.ego_vels = dataset.ego_vels
        self.commands = dataset.commands

        del dataset
        print(f"Loaded MVAdapt Dataset from {path}")
        
    def merge(self, dataset):
        self.vehicle_indices.extend(dataset.vehicle_indices)
        self.gt_waypoints.extend(dataset.gt_waypoints)
        self.bs_waypoints.extend(dataset.bs_waypoints)
        self.gt_controls.extend(dataset.gt_controls)
        self.bs_controls.extend(dataset.bs_controls)
        self.scene_features.extend(dataset.scene_features)
        self.physics_params.extend(dataset.physics_params)
        self.gear_params.extend(dataset.gear_params)

        self.rgb_paths.extend(dataset.rgb_paths)
        self.lidar_paths.extend(dataset.lidar_paths)
        self.target_points.extend(dataset.target_points)
        self.ego_vels.extend(dataset.ego_vels)
        self.commands.extend(dataset.commands)
        
    def load_physics_data(self, vehicle_index, norm=True):
        item = self.vehicle_config.config_list[vehicle_index]
        
        physics_prop = []
        gear_prop = []

        wheelbase = calculate_wheelbase(item)
        if norm:
            item["vehicle_extent"] = normalize(item["vehicle_extent"], self.config.max_extent, self.config.min_extent)
            item["physics"]["torque_curve"] = normalize(pad_data(item["physics"]["torque_curve"], 2 * self.config.max_torque_num), self.config.max_torque_curve, self.config.min_torque_curve)
            item["physics"]["max_rpm"] = normalize(item["physics"]["max_rpm"], self.config.max_max_rpm, self.config.min_max_rpm)
            item["physics"]["wheels"][0]["radius"] = normalize(item["physics"]["wheels"][0]["radius"], self.config.max_radius, self.config.min_radius)
            item["physics"]["mass"] = normalize(item["physics"]["mass"], self.config.max_mass, self.config.min_mass)
            wheelbase = normalize(wheelbase, self.config.max_wheelbase, self.config.min_wheelbase)

        physics_prop.extend(item["vehicle_extent"])
        physics_prop.extend(item["physics"]["torque_curve"])
        physics_prop.append(item["physics"]["max_rpm"])
        physics_prop.append(item["physics"]["wheels"][0]["radius"])
        physics_prop.extend(list(item["physics"]["center_of_mass"]))
        physics_prop.append(item["physics"]["mass"])
        physics_prop.append(wheelbase)

        for gear in item["physics"]["forward_gears"]:
            gear_prop.extend([gear["ratio"], gear["up_ratio"], gear["down_ratio"]])
        gear_prop = pad_data(gear_prop, 3 * self.config.max_gear_num)

        return np.array(physics_prop, dtype=float), np.array(gear_prop, dtype=float)

    def initialize(self, args, split='train', clear_crashed=False, clear_imperfect=False, move_dup_dir=None):
        print("Initializing MVAdapt Dataset...")
        print(f"Root Directory: {self.root_dir}")
        
        if move_dup_dir is not None:
            print(f"Moving duplicated data to {move_dup_dir}")
            move_duplicated_data(self.root_dir, move_dup_dir)
            print("Moving duplicated data done.")

        agent = BasemodelAgent(args.base_model, verbose=args.verbose)
        self.config = agent.config

        physics_cache = {}

        if args.vehicle_indices == 'all':
            vehicle_list = all_vehicles
        else:
            vehicle_list = eval(args.vehicle_indices)

        for v_index in vehicle_list:
            self.config.update_vehicle(v_index)
            self.config.initialize(root_dir=self.root_dir, vehicle_index=v_index, verbose=args.verbose)

            data_dir = self.config.train_data if split == 'train' else self.config.val_data
            shuffle = True if split == 'train' else False
            carla_set = CARLA_Data(data_dir, self.config, verbose=args.verbose, clear_crashed=clear_crashed, clear_imperfect=clear_imperfect)
            dataloader = DataLoader(carla_set, batch_size=args.process_batch, num_workers=6, shuffle=shuffle, pin_memory=True)

            if v_index not in physics_cache:
                physics_cache[v_index] = self.load_physics_data(v_index)
            physics_prop, gear_prop = physics_cache[v_index]

            for data in tqdm(dataloader, desc=f"Processing {split.capitalize()} Data - Vehicle {v_index}"):
                batch_size = data['rgb'].shape[0]
                result = process_data(data, agent, args.device, self.root_dir)

                self.vehicle_indices.extend([v_index] * batch_size)
                self.gt_waypoints.extend(result[0])
                self.bs_waypoints.extend(result[1])
                self.gt_controls.extend(result[2])
                self.bs_controls.extend(result[3])
                self.scene_features.extend(result[4])
                self.physics_params.extend([physics_prop] * batch_size)
                self.gear_params.extend([gear_prop] * batch_size)
                self.rgb_paths.extend(result[5])
                self.lidar_paths.extend(result[6])
                self.target_points.extend(result[7])
                self.ego_vels.extend(result[8])
                self.commands.extend(result[9])

                torch.cuda.empty_cache()
                del result
            try:
                self.save(f"/home/ohs-dyros/gitRepo/MVAdapt/dataset/checkpoint_{v_index}.pkl")
                print(f"Saved checkpoint for vehicle {v_index}")
            except Exception as e:
                print(f"Failed to save checkpoint.: {e}")
                
                
    def sample_data_per_vehicle(self, num_samples, exclude=[]):
        v2index = defaultdict(list)
        for idx, v_index in enumerate(self.vehicle_indices):
            if v_index not in exclude:
                v2index[v_index].append(idx)
        
        sampled_indices = []
        for v_index, indices in v2index.items():
            if len(indices) > num_samples:
                sampled_indices.extend(random.sample(indices, num_samples))
            else:
                sampled_indices.extend(indices) 
        
        dataset = MVAdaptDataset(self.root_dir, self.config)
        for idx in sampled_indices:
            dataset.vehicle_indices.append(self.vehicle_indices[idx])
            dataset.gt_waypoints.append(self.gt_waypoints[idx])
            dataset.bs_waypoints.append(self.bs_waypoints[idx])
            dataset.gt_controls.append(self.gt_controls[idx])
            dataset.bs_controls.append(self.bs_controls[idx])
            dataset.scene_features.append(self.scene_features[idx])
            dataset.physics_params.append(self.physics_params[idx])
            dataset.gear_params.append(self.gear_params[idx])
            dataset.rgb_paths.append(self.rgb_paths[idx])
            dataset.lidar_paths.append(self.lidar_paths[idx])
            dataset.target_points.append(self.target_points[idx])
            dataset.ego_vels.append(self.ego_vels[idx])
            dataset.commands.append(self.commands[idx])
        
        return dataset
