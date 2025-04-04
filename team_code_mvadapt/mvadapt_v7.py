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
            nn.ELU(),
            nn.Linear(latent_dim, output_dim),
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
            nn.ELU(),
            nn.Linear(64, latent_dim),
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
    
class ImageTransformerEncoder(nn.Module):
    def __init__(self, img_size_w, img_size_h, out_dim, patch_size=32, in_channels=3, emb_dim=256, num_layers=6, num_heads=8):
        super(ImageTransformerEncoder, self).__init__()
        H, W = img_size_w, img_size_h
        self.P = patch_size
        self.num_patches = (H // self.P) * (W // self.P)  
        self.patch_dim = self.P * self.P * in_channels  

        self.proj = nn.Linear(self.patch_dim, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.P 

        x = x.unfold(2, P, P).unfold(3, P, P)  
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B, -1, P * P * C) 
        x = self.proj(x) 
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.transformer(x) 
        cls_embedding = x[:, 0, :]
        return self.mlp(cls_embedding)
    
class PhysicsAttentionEncoder(nn.Module):
    def __init__(self, scene_dim, physics_dim, target_points=8):
        super(PhysicsAttentionEncoder, self).__init__()
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, scene_dim),
            nn.ELU()
        )
        self.physics_attention = nn.Sequential(
            nn.Linear(physics_dim, physics_dim),
            nn.LayerNorm(physics_dim),
            nn.ELU(),
            nn.Linear(physics_dim, scene_dim),
            nn.Tanh(),
            nn.Softmax()
        )
        self.target_points = target_points
    
    def forward(self, scene_embedding, physics_embedding):
        bs = scene_embedding.size(0)
        scene_embedding = self.scene_encoder(scene_embedding)
        physics_attention = self.physics_attention(physics_embedding).reshape(bs, 1, -1).repeat(1, self.target_points, 1)
        scene_embedding = scene_embedding * physics_attention
        return scene_embedding
    
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
        self.latent_dim = 128
        self.gear_dim = 4
        self.image_width = 1024
        self.image_height = 256
        self.image_embedding_dim = 256
        
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
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(self.device)
        
        # self.transformer_encoder = CrossAttentionTransformer(scene_dim=self.input_dim, physics_dim=self.latent_dim).to(self.device)
        self.physics_attention_encoder = PhysicsAttentionEncoder(scene_dim=self.input_dim, physics_dim=self.latent_dim).to(self.device)
        
        self.image_attention_encoder = ImageTransformerEncoder(
            img_size_w=self.image_width,
            img_size_h=self.image_height,
            out_dim=self.image_embedding_dim
        ).to(self.device)
        
        self.core_net = nn.Sequential(
            nn.Linear(self.input_dim + self.image_embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.input_dim),
        ).to(self.device)
        
        self.decoder = nn.Linear(self.hidden_size, 2).to(self.device)

    def forward(self, rgb, scene_feature, target_point, physics_params, gear_params):
        if scene_feature.dim() == 1:
            scene_feature = scene_feature.unsqueeze(0)
        if scene_feature.size()[-1] != 2:
            scene_feature = scene_feature.reshape(-1, self.waypoints, self.input_dim)

        bs = scene_feature.size(0)
        
        if rgb.dim() == 1:
            rgb = rgb.unsqueeze(0)
        
        if target_point.dim() == 1:
            target_point = target_point.unsqueeze(0)
        
        if physics_params.dim() == 1:
            physics_params = physics_params.unsqueeze(0)
        
        if gear_params.dim() == 1:
            gear_params = gear_params.unsqueeze(0)

        physics_latent = self.physics_encoder(physics_params, gear_params).unsqueeze(0)
        combined_scene = self.physics_attention_encoder(scene_feature, physics_latent)
        
        image_embedding = self.image_attention_encoder(rgb).reshape(bs, 1, -1).repeat(1, self.waypoints, 1)
        combined_scene = torch.cat((combined_scene, image_embedding), dim=2)
        combined_scene = self.core_net(combined_scene)
        
        z = self.encoder(target_point).unsqueeze(0)
        
        output, _ = self.gru(combined_scene, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)    

        return output.reshape(-1, self.waypoints * 2)

    def inference(self, rgb, scene_feature, target_point, physics_params, gear_params):
        self.eval()
        with torch.no_grad():
            return self.forward(rgb, scene_feature, target_point, physics_params, gear_params)
        
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