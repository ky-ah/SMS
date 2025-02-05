import torch
import torch.nn as nn
from einops import reduce
from models.networks import FiLM, CLIP_ResNet


class FiLMedModel(nn.Module):
    def __init__(self, cfg, comp_struct):
        super().__init__()

        # Vision-language model
        self.encoder = CLIP_ResNet()

        # Hidden dimension 
        self.emb_dim = self.encoder.output_size

        # All-shared strategy
        self.all_shared = cfg.strategy == "all-shared"

        # FiLM modules
        self.controllers = []
        for i in range(4):
            mod = FiLM(
                in_features=self.emb_dim,
                in_channels=self.emb_dim,
                config=comp_struct[i],
                T = cfg.T
            )
            self.controllers.append(mod)
            self.add_module("FiLM_Layer_" + str(i), mod)

        # Summary of task-specific and shared submodules
        self.shared = nn.ModuleList([c.shared for c in self.controllers])
        self.specialized = nn.ModuleList([c.task_specific for c in self.controllers])

        # Add final projection layer to self.shared or self.specialized
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
        l = self.encoder(mission, text=True)

        for controller in self.controllers:
            z1 = controller(z1, l, task)

        z1, z2, z3 = [reduce(z, "b c h w -> b c", "max") for z in [z1, z2, z3]]
        
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
