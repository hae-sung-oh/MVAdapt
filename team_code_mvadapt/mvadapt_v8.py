import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_transformer_encoder import MaskedTransformerEncoder
from mvadapt_v4 import PhysicsEncoder

class SimpleBodyTransformer(nn.Module):
    def __init__(self, nbodies, input_dim, dim_feedforward=256, nhead=4, num_layers=3, use_positional_encoding=True):
        super().__init__()
        self.nbodies = nbodies
        self.input_dim = input_dim
        self.use_positional_encoding = use_positional_encoding

        assert input_dim % nhead == 0, "input_dim must be divisible by nhead"

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = MaskedTransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_positional_encoding:
            self.position_embedding = nn.Embedding(nbodies, input_dim)

        adjacency_matrix = torch.ones(nbodies, nbodies, dtype=torch.bool)
        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, x):
        if self.use_positional_encoding:
            indices = torch.arange(self.nbodies, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
            x = x + self.position_embedding(indices)

        x = self.encoder(x, mask=~self.adjacency_matrix)
        return x

class MVAdapt(nn.Module):
    def __init__(self, config, args=None, device='cuda:0'):
        super(MVAdapt, self).__init__()
        self.config = config
        self.device = device

        self.input_dim = config.gru_input_size  # 256
        self.hidden_size = config.gru_hidden_size
        self.waypoints = config.pred_len // config.wp_dilation

        self.physics_dim = 18
        self.gear_length = 27
        self.latent_dim = 128
        self.gear_dim = 4
        self.nbodies = 8

        self.encoder = nn.Linear(2, self.hidden_size).to(device)

        self.physics_encoder = PhysicsEncoder(
            input_dim=self.physics_dim,
            latent_dim=self.latent_dim,
            gear_length=self.gear_length,
            gear_dim=self.gear_dim,
            device=device
        ).to(device)

        self.project_to_tokens = nn.Linear(self.latent_dim, self.nbodies * self.latent_dim).to(device)
        self.scene_proj_layer = nn.Linear(self.input_dim, self.latent_dim).to(device)

        self.body_transformer = SimpleBodyTransformer(
            nbodies=self.nbodies + self.waypoints,
            input_dim=self.latent_dim,
            dim_feedforward=256,
            nhead=4,
            num_layers=3,
            use_positional_encoding=True
        ).to(device)

        self.gru = nn.GRU(input_size=self.latent_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True).to(device)
        self.decoder = nn.Linear(self.hidden_size, 2).to(device)

    def forward(self, rgb, scene_feature, target_point, physics_params, gear_params):
        if scene_feature.dim() == 1:
            scene_feature = scene_feature.unsqueeze(0)
        if scene_feature.size(-1) != self.input_dim:
            scene_feature = scene_feature.view(-1, self.nbodies, self.input_dim)

        bs = scene_feature.size(0)

        physics_latent = self.physics_encoder(physics_params, gear_params)
        tokens = self.project_to_tokens(physics_latent).view(bs, self.nbodies, self.latent_dim)

        scene_feature = scene_feature.view(bs, self.waypoints, self.input_dim)
        scene_proj = self.scene_proj_layer(scene_feature)

        tokens = torch.cat([tokens, scene_proj], dim=1)

        transformer_out = self.body_transformer(tokens)
        fused = transformer_out[:, self.nbodies:, :]

        z = self.encoder(target_point).unsqueeze(0)
        fused = fused.contiguous()
        output, _ = self.gru(fused, z)

        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, dim=1)
        return output.view(bs, -1)

    def inference(self, rgb, scene_feature, target_point, physics_params, gear_params):
        self.eval()
        with torch.no_grad():
            return self.forward(rgb, scene_feature, target_point, physics_params, gear_params)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cuda:0'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
