import torch
import torch.nn as nn
import torch.nn.functional as F
import carla

from basemodel_agent import BasemodelAgent

class ForwardGearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, device='cuda:0'):
        super(ForwardGearEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
            nn.Dropout(0.2),
        ).to(device)
        
    def forward(self, x):
        return self.fc(x)
    
class PhysicsEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, gear_length, gear_dim, device='cuda:0'):
        super(PhysicsEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.gear_encoder = ForwardGearEncoder(input_dim=gear_length, latent_dim=latent_dim, output_dim=gear_dim).to(device)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + gear_dim, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Dropout(0.2),
        )
    
    def forward(self, physics_params, gear_params):
        physics_params = physics_params.to(self.device)
        gear_latent = self.gear_encoder(gear_params)
        if physics_params.dim() == 1:
            physics_params = physics_params.unsqueeze(0)
        if gear_latent.dim() == 1:
            gear_latent = gear_latent.unsqueeze(0)
        physics_params = torch.cat((physics_params, gear_latent), dim=1)
        return self.encoder(physics_params)
    
class MVAdapt(nn.Module):
    def __init__(self, config, args=None, device='cuda:0'):
        super(MVAdapt, self).__init__()
        self.config = config
        self.device = device
        self.input_dim = self.config.gru_input_size
        self.hidden_size = self.config.gru_hidden_size
        self.waypoints = self.config.pred_len // self.config.wp_dilation
        self.physics_dim = 18
        self.gear_length = 27
        self.latent_dim = 32
        self.gear_dim = 4
        
        self.encoder = nn.Linear(2, self.hidden_size).to(self.device)
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True).to(self.device)
        self.physics_encoder = PhysicsEncoder(
            input_dim=self.physics_dim,
            latent_dim=self.latent_dim,
            gear_length=self.gear_length,
            gear_dim=self.gear_dim
        ).to(self.device)
        self.physics_hidden = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.2),
        ).to(self.device)
        
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.2),
        ).to(self.device)

        self.decoder = nn.Linear(self.hidden_size * 2, 2).to(self.device)

    def forward(self, x, target_point, physics_params, gear_params):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size()[-1] != 2:
            x = x.reshape(-1, 8, 256)

        bs = x.size(0)

        physics_latent = self.physics_encoder(physics_params, gear_params)
        physics_latent = self.physics_hidden(physics_latent)

        combined_latent = torch.cat([self.encoder(target_point).unsqueeze(0), physics_latent.unsqueeze(0)], dim=-1)
        z = self.fusion_encoder(combined_latent)

        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        physics_expanded = physics_latent.unsqueeze(1).repeat(1, self.waypoints, 1).reshape(bs * self.waypoints, -1)
        output = torch.cat([output, physics_expanded], dim=-1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)    

        return output.reshape(-1, 8 * 2)

    def inference(self, x, target_point, physics_params, gear_params):
        self.eval()
        with torch.no_grad():
            return self.forward(x, target_point, physics_params, gear_params)
        
    def debug(self, path, baseline_path, dataset, max_num=None):
        agent = BasemodelAgent(baseline_path)
        if max_num is None:
            max_num = len(dataset)
        
        for i in range(max_num):
            data = dataset[i]
            rgb = torch.tensor(data['rgb'], dtype=torch.float32).cuda().unsqueeze(0)
            lidar = torch.tensor(data['lidar'], dtype=torch.float32).cuda().unsqueeze(0)
            target_point = torch.tensor(data['target_point'], dtype=torch.float32).cuda().unsqueeze(0)
            ego_vel = torch.tensor(data['ego_vel'], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(1)
            command = torch.tensor(data['command'], dtype=torch.float32).cuda().unsqueeze(0)

            pred_wp, \
            pred_target_speed, \
            pred_checkpoint, \
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            _, _, _, \
            joined_wp_features = agent.nets[0].forward(
                rgb=rgb,
                lidar_bev=lidar,
                target_point=target_point,
                ego_vel=ego_vel,
                command=command)

            agent.nets[0].visualize_model(
                path,
                i,
                rgb,
                lidar,
                target_point,
                pred_wp,
                pred_semantic=pred_semantic,
                pred_bev_semantic=pred_bev_semantic,
                pred_depth=pred_depth,
                pred_checkpoint=pred_checkpoint,
                pred_speed=pred_target_speed,
                gt_wp=torch.tensor(data['gt_waypoint'].reshape(-1, 2), dtype=torch.float32).cuda().unsqueeze(0),
                mv_wp=self.inference(
                    joined_wp_features,
                    target_point,
                    torch.tensor(data['physics_params'], dtype=torch.float32).cuda().unsqueeze(0),
                    torch.tensor(data['gear_params'], dtype=torch.float32).cuda().unsqueeze(0)
                ).reshape(-1, 2).unsqueeze(0),
            )
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cuda:0'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)