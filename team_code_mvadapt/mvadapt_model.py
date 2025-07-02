import torch
import torch.nn as nn

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
    
class CrossAttentionTransformer(nn.Module):
    def __init__(self, scene_dim, physics_dim, num_heads=8, ff_dim=512, target_points=8):
        super().__init__()
        self.target_points = target_points
        self.scene_dim = scene_dim
        self.scene_q = nn.Linear(scene_dim, scene_dim)
        self.physics_k = nn.Linear(physics_dim, scene_dim) 
        self.scene_v = nn.Linear(scene_dim, scene_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=scene_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(scene_dim, ff_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(ff_dim, scene_dim),
            nn.Dropout(0.2),
        )
        
        self.norm1 = nn.LayerNorm(scene_dim)
        self.norm2 = nn.LayerNorm(scene_dim)
    
    def forward(self, scene_embedding, physics_embedding):
        bs = scene_embedding.size(0)
        scene_q = self.scene_q(scene_embedding)
        physics_k = self.physics_k(physics_embedding).reshape(bs, -1, self.scene_dim).repeat(1, self.target_points, 1)
        scene_v = self.scene_v(scene_embedding)
    
        adapted_scene, _ = self.cross_attention(scene_q, physics_k, scene_v)
        scene_embedding = self.norm1(scene_q + adapted_scene) 
        ff_output = self.feed_forward(scene_embedding)
        output = self.norm2(scene_embedding + ff_output)
        
        return output
    
class MultiLayerCrossAttention(nn.Module):
    def __init__(self, scene_dim, physics_dim, num_layers=4, num_heads=8, ff_dim=512, target_points=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionTransformer(
                scene_dim=scene_dim,
                physics_dim=physics_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                target_points=target_points
            ) for _ in range(num_layers)
        ])
    
    def forward(self, scene_embedding, physics_embedding):
        for layer in self.layers:
            scene_embedding = layer(scene_embedding, physics_embedding)
        return scene_embedding
    
class MVAdapt(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super(MVAdapt, self).__init__()
        self.config = config
        self.device = device
        self.input_dim = self.config.gru_input_size
        self.hidden_size = self.config.gru_hidden_size
        self.waypoints = self.config.pred_len // self.config.wp_dilation
        self.physics_dim = self.config.physics_dim
        self.gear_length = self.config.gear_length
        self.latent_dim = self.config.mvadapt_latent_dim
        self.gear_dim = self.config.mvadapt_gear_dim
        
        self.encoder = nn.Linear(2, self.hidden_size).to(self.device)
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True).to(self.device)
        
        self.physics_encoder = PhysicsEncoder(
            input_dim=self.physics_dim,
            latent_dim=self.latent_dim,
            gear_length=self.gear_length,
            gear_dim=self.gear_dim
        ).to(self.device)
        
        self.transformer_encoder = MultiLayerCrossAttention(
            scene_dim=self.input_dim,
            physics_dim=self.latent_dim,
            num_layers=4 
        ).to(self.device)
        self.decoder = nn.Linear(self.hidden_size, 2).to(self.device)

    def forward(self, scene_feature, target_point, physics_params, gear_params):
        if scene_feature.dim() == 1:
            scene_feature = scene_feature.unsqueeze(0)
        if scene_feature.size()[-1] != 2:
            scene_feature = scene_feature.reshape(-1, self.waypoints, self.input_dim)

        bs = scene_feature.size(0)
        
        if target_point.dim() == 1:
            target_point = target_point.unsqueeze(0)
        
        if physics_params.dim() == 1:
            physics_params = physics_params.unsqueeze(0)
        
        if gear_params.dim() == 1:
            gear_params = gear_params.unsqueeze(0)

        physics_embedding = self.physics_encoder(physics_params, gear_params).unsqueeze(1)
        combined_scene = self.transformer_encoder(scene_feature, physics_embedding)
        
        z = self.encoder(target_point).unsqueeze(0)
        
        output, _ = self.gru(combined_scene, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)    

        return output.reshape(-1, self.waypoints * 2)

    def inference(self, scene_feature, target_point, physics_params, gear_params):
        self.eval()
        with torch.no_grad():
            return self.forward(scene_feature, target_point, physics_params, gear_params)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cuda:0'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)