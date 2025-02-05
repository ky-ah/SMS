import torch
import torch.nn as nn
from einops import rearrange, reduce
from models.networks import TransformerEncoderLayer, CLIP_ResNet, CLIP_ViT


class AttentionModel(nn.Module):
    def __init__(self, cfg, comp_struct):
        super().__init__()

        # Vision-language model
        self.encoder = CLIP_ResNet()

        # Hidden dimension 
        self.emb_dim = self.encoder.output_size

        # All-shared strategy
        self.all_shared = cfg.strategy == "all-shared"

        # Transformer Encoder Layers
        self.attention = []
        for i in range(4):
            t_layer = TransformerEncoderLayer(
                self.emb_dim, dim_feedforward=64, config=comp_struct[i], T=cfg.T
            )
            self.attention.append(t_layer)
            self.add_module("Transformer_Layer_" + str(i), t_layer)

        # Summary of task-specific and shared submodules
        self.shared = nn.ModuleList([a.shared for a in self.attention])
        self.specialized = nn.ModuleList([a.task_specific for a in self.attention])

         # Add final projection layer to self.shared or self.specialized based on comp_struct
        if self.all_shared:
            self.project = nn.Linear(self.emb_dim, self.emb_dim)
            self.shared.append(self.project)
        else:
            self.project = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for _ in range(cfg.T)])
            self.specialized.append(self.project)

    def forward(self, x1, x2, x3, mission, task=None):
        """
        Input:
            x1: anchor/base image
            x2: positive implication image
            x3: negative implication image
        Output:
            p1: language-conditioned anchor image encoding
            z2: positive implication image encoding
            z3: negative implication image encoding
        """
        z1, z2, z3 = [self.encoder(x) for x in [x1, x2, x3]]
        z1, z2, z3 = [rearrange(z, "b c h w -> b (h w) c") for z in [z1, z2, z3]]

        l = self.encoder(mission, text=True).unsqueeze(1)

        z1 = torch.cat((z1, l), dim=1)
        for t_layer in self.attention:
            z1 = t_layer(z1, task)

        z1, z2, z3 = [reduce(z, "b hw c -> b c", "max") for z in [z1, z2, z3]]
        
        if self.all_shared:
            p1, z2, z3 = [self.project(z) for z in [z1, z2, z3]]
        else:
            if isinstance(task, int):
                p1, z2, z3 = [self.project[task](z) for z in [z1, z2, z3]]
            else:
                p1 = torch.stack([self.project[t](z) for t, z in zip(task, z1)])
                z2 = torch.stack([self.project[t](z) for t, z in zip(task, z2)])
                z3 = torch.stack([self.project[t](z) for t, z in zip(task, z3)])

        return p1, z2, z3
    

class AllAttentionModel(nn.Module):
    def __init__(self, cfg, comp_struct):
        super().__init__()

        # Vision-language model
        self.encoder = CLIP_ViT()

        # Hidden dimension 
        self.emb_dim = self.encoder.output_size

        # All-shared strategy
        self.all_shared = cfg.strategy == "all-shared"

        # Transformer Encoder Layers
        self.attention = []
        for i in range(4):
            t_layer = TransformerEncoderLayer(
                self.emb_dim, dim_feedforward=64, config=comp_struct[i], T=cfg.T
            )
            self.attention.append(t_layer)
            self.add_module("Transformer_Layer_" + str(i), t_layer)

        # Summary of task-specific and shared submodules
        self.shared = nn.ModuleList([a.shared for a in self.attention])
        self.specialized = nn.ModuleList([a.task_specific for a in self.attention])

         # Add final projection layer to self.shared or self.specialized based on comp_struct
        if self.all_shared:
            self.project = nn.Linear(self.emb_dim, self.emb_dim)
            self.shared.append(self.project)
        else:
            self.project = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for _ in range(cfg.T)])
            self.specialized.append(self.project)

    def forward(self, x1, x2, x3, mission, task=None):
        """
        Input:
            x1: anchor/base image
            x2: positive implication image
            x3: negative implication image
        Output:
            p1: language-conditioned anchor image encoding
            z2: positive implication image encoding
            z3: negative implication image encoding
        """
        z1, z2, z3 = [self.encoder(x) for x in [x1, x2, x3]]

        l = self.encoder(mission, text=True).unsqueeze(1)

        z1 = torch.cat((z1, l), dim=1)
        for t_layer in self.attention:
            z1 = t_layer(z1, task)
        
        if self.all_shared:
            p1, z2, z3 = [self.project(z[:, 0, :]) for z in [z1, z2, z3]]
        else:
            if isinstance(task, int):
                p1, z2, z3 = [self.project[task](z[:, 0, :]) for z in [z1, z2, z3]]
            else:
                p1 = torch.stack([self.project[t](z[:, 0, :]) for t, z in zip(task, z1)])
                z2 = torch.stack([self.project[t](z[:, 0, :]) for t, z in zip(task, z2)])
                z3 = torch.stack([self.project[t](z[:, 0, :]) for t, z in zip(task, z3)])

        return p1, z2, z3
