from team_code.vehicle_config import VehicleConfig
from team_code.config import GlobalConfig
import numpy as np
import torch 

def normalize(value, max, min):
    if isinstance(value, (list, tuple)):
        return type(value)(normalize(v, max, min) for v in value)
    else:
        return (value - min) / (max - min)
    
class MVAdaptDataLoader:
    def __init__(self, physics_norm=True):
        self.v_config = VehicleConfig()
        self.g_config = GlobalConfig()
        self.norm = physics_norm

    def load_physics_data(self, vehicle_index, device='gpu:0'):
        physics_prop = []
        gear_prop = []

        item = self.v_config.config_list[vehicle_index]
        if self.norm:
            item["vehicle_extent"] = normalize(item["vehicle_extent"], self.g_config.max_extent, self.g_config.min_extent)
            item["physics"]["torque_curve"] = normalize(item["physics"]["torque_curve"], self.g_config.max_torque_curve, self.g_config.min_torque_curve)
            item["physics"]["max_rpm"] = normalize(item["physics"]["max_rpm"], self.g_config.max_max_rpm, self.g_config.min_max_rpm)
            item["physics"]["wheels"][0]["radius"] = normalize(item["physics"]["wheels"][0]["radius"], self.g_config.max_radius, self.g_config.min_radius)
            item["physics"]["mass"] = normalize(item["physics"]["mass"], self.g_config.max_mass, self.g_config.min_mass)
            
        physics_prop.extend(item["vehicle_extent"])
        physics_prop.extend([i for tup in item["physics"]["torque_curve"] for i in tup])
        physics_prop.append(item["physics"]["max_rpm"])
        physics_prop.append(item["physics"]["wheels"][0]["radius"])
        physics_prop.extend(list(item["physics"]["center_of_mass"]))
        physics_prop.append(item["physics"]["mass"])
        for gear in item["physics"]["forward_gears"]:
            gear_prop.append([gear["ratio"], gear["up_ratio"], gear["down_ratio"]])

        physics_prop = np.array(physics_prop)
        gear_prop = np.array(gear_prop)
        physics_prop = torch.tensor(physics_prop, dtype=torch.float32).to(device)
        gear_prop = torch.tensor(gear_prop, dtype=torch.float32).to(device)
        
        return physics_prop, gear_prop

