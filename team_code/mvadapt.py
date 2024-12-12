import torch
import torch.nn as nn
import torch.nn.functional as F
import carla

class FowardGearEncoder(nn.Module):
    def __init__(self, max_length, embed_dim=16, num_heads=4, ff_dim=64, num_layers=2, output_dim=4):
        self.max_length = max_length
        super(FowardGearEncoder, self).__init__()
        self.embedding = nn.Linear(3, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)  


    def forward(self, x):
        padded_data = torch.zeros((self.max_length, 3))
        mask = torch.zeros((self.max_length))
        for i, gear in enumerate(x):
            padded_data[i] = gear
            mask[i] = 1
        x = padded_data
        x = self.embedding(x) + self.positional_encoding
        key_padding_mask = (mask == 0).reshape(1, -1)
        x = self.transformer(x.permute(1, 0, 2), src_key_padding_mask=key_padding_mask)
        x = torch.mean(x, dim=0)
        return self.fc(x)[0]

class PhysicsEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, gear_max_length=8, gear_dim=4):
        super(PhysicsEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gear_encoder = FowardGearEncoder(max_length=gear_max_length, output_dim=gear_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + gear_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
    
    def forward(self, physics_params, gear_params):
        if physics_params.shape[0] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {physics_params.shape[0]}")
        # physics_params = physics_params.to('cuda')
        gear_latent = self.gear_encoder(gear_params)
        physics_params = torch.cat((physics_params, gear_latent), dim=0)
        return self.encoder(physics_params)

class MVAdapt(nn.Module):
    def __init__(self, layer_dim0, layer_dim1, physics_input_dim=15, gear_max_length=8, gear_dim=4):
        super(MVAdapt, self).__init__()
        self.layer_dim0 = layer_dim0
        self.layer_dim1 = layer_dim1
        physics_latent_dim = 3 * layer_dim0 + layer_dim0 * layer_dim1 + layer_dim1 * 3
        self.physics_encoder = PhysicsEncoder(physics_input_dim, physics_latent_dim, gear_max_length, gear_dim)
        
        self.layer0 = nn.Linear(3, layer_dim0)
        self.layer1 = nn.Linear(layer_dim0, layer_dim1)
        self.layer2 = nn.Linear(layer_dim1, 3)
        
        self.adaptation_module = nn.Sequential(
            self.layer0, 
            nn.ReLU(),
            self.layer1,
            nn.ReLU(),
            self.layer2,
        )
    
    def forward(self, control_input, physics_params, gear_params):
        weights = self.physics_encoder(physics_params, gear_params)
        control_input = torch.tensor([[control_input.throttle, control_input.steer, control_input.brake]])
        
        weights0 = weights[:3 * self.layer_dim0].view(self.layer_dim0, 3)
        weights1 = weights[3 * self.layer_dim0: 3 * self.layer_dim0 + self.layer_dim0 * self.layer_dim1].view(self.layer_dim1, self.layer_dim0)
        weights2 = weights[3 * self.layer_dim0 + self.layer_dim0 * self.layer_dim1:].view(3, self.layer_dim1)
        biases0 = torch.zeros(self.layer_dim0)
        biases1 = torch.zeros(self.layer_dim1)
        biases2 = torch.zeros(3)
        
        with torch.no_grad():
            self.layer0.weight = nn.Parameter(weights0)
            self.layer0.bias = nn.Parameter(biases0)
            self.layer1.weight = nn.Parameter(weights1)
            self.layer1.bias = nn.Parameter(biases1)
            self.layer2.weight = nn.Parameter(weights2)
            self.layer2.bias = nn.Parameter(biases2)
        
        output = self.adaptation_module(control_input)
        
        output = carla.VehicleControl( # type: ignore
            throttle=float(torch.sigmoid(output[:, 0])), 
            steer=float(torch.tanh(output[:, 1])), 
            brake=float(torch.sigmoid(output[:, 2]))
        )
        return output
