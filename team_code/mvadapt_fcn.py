import torch
import torch.nn as nn
import torch.nn.functional as F
import carla

class ForwardGearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, device='cuda:0'):
        super(ForwardGearEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
            nn.ReLU(),
        ).to(device)
        
    def forward(self, x):
        return self.fc(x)
    
class PhysicsEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, gear_max_length=24, gear_dim=4, device='cuda:0'):
        super(PhysicsEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.gear_encoder = ForwardGearEncoder(input_dim=gear_max_length, latent_dim=8, output_dim=gear_dim).to(device)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + gear_dim, 64),
            nn.ReLU(),
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
class MVAdapt(nn.Module):
    def __init__(self, layer_dim0, layer_dim1, layer_dim2, layer_dim3, physics_input_dim=18, gear_max_length=24, gear_dim=4, device='cuda:0'):
        super(MVAdapt, self).__init__()
        self.layer_dim0 = layer_dim0
        self.layer_dim1 = layer_dim1
        self.layer_dim2 = layer_dim2
        self.layer_dim3 = layer_dim3
        self.device = device
        
        physics_latent_dim = (
            16 * layer_dim0 + layer_dim0 + 
            layer_dim0 * layer_dim1 + layer_dim1 + 
            layer_dim1 * layer_dim2 + layer_dim2 +
            layer_dim2 * layer_dim3 + layer_dim3 +
            16 * layer_dim3 + 16
        )
        
        self.physics_encoder = PhysicsEncoder(
            input_dim=physics_input_dim, 
            latent_dim=physics_latent_dim, 
            gear_max_length=gear_max_length, 
            gear_dim=gear_dim
        ).to(device)

    def forward(self, waypoint_input, physics_params, gear_params):
        if waypoint_input.dim() == 1:
            waypoint_input = waypoint_input.unsqueeze(0)

        batch_size = waypoint_input.size(0)
        
        weights = self.physics_encoder(physics_params, gear_params).to(self.device)
        
        index0 = 16 * self.layer_dim0
        index1 = index0 + self.layer_dim0
        index2 = index1 + self.layer_dim0 * self.layer_dim1
        index3 = index2 + self.layer_dim1
        index4 = index3 + self.layer_dim1 * self.layer_dim2
        index5 = index4 + self.layer_dim2
        index6 = index5 + self.layer_dim2 * self.layer_dim3
        index7 = index6 + self.layer_dim3
        index8 = index7 + 16 * self.layer_dim3
        
        weights0 = weights[:, :index0].view(batch_size, self.layer_dim0, 16).to(self.device)
        biases0 = weights[:, index0:index1].view(batch_size, self.layer_dim0).to(self.device)
        weights1 = weights[:, index1:index2].view(batch_size, self.layer_dim1, self.layer_dim0).to(self.device)
        biases1 = weights[:, index2:index3].view(batch_size, self.layer_dim1).to(self.device)
        weights2 = weights[:, index3:index4].view(batch_size, self.layer_dim2, self.layer_dim1).to(self.device)
        biases2 = weights[:, index4:index5].view(batch_size, self.layer_dim2).to(self.device)
        weights3 = weights[:, index5:index6].view(batch_size, self.layer_dim3, self.layer_dim2).to(self.device)
        biases3 = weights[:, index6:index7].view(batch_size, self.layer_dim3).to(self.device)
        weights4 = weights[:, index7:index8].view(batch_size, 16, self.layer_dim3).to(self.device)
        biases4 = weights[:, index8:].view(batch_size, 16).to(self.device)
        
        outputs = []
        for i in range(batch_size):
            x = torch.nn.functional.linear(waypoint_input[i], weights0[i], biases0[i])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights1[i], biases1[i])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights2[i], biases2[i])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights3[i], biases3[i])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights4[i], biases4[i])
            outputs.append(x)

        outputs = torch.stack(outputs)
        return outputs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cuda:0'):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.to(device)
