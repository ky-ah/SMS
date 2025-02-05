from collections import OrderedDict
from einops import rearrange, reduce
import torch
from torch import nn
import torch.nn.functional as F
import clip

class CLIP_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, _ = clip.load("RN50")
        self._output_size = 1024

    def forward(self, x, text=False):
        if text:
            x = x.squeeze(1)
            return self.encoder.encode_text(x)
        else:
            def stem(x):
                x = self.encoder.visual.relu1(self.encoder.visual.bn1(self.encoder.visual.conv1(x)))
                x = self.encoder.visual.relu2(self.encoder.visual.bn2(self.encoder.visual.conv2(x)))
                x = self.encoder.visual.relu3(self.encoder.visual.bn3(self.encoder.visual.conv3(x)))
                x = self.encoder.visual.avgpool(x)
                return x

            x = x.type(self.encoder.visual.conv1.weight.dtype)
            x = stem(x)
            x = self.encoder.visual.layer1(x)
            x = self.encoder.visual.layer2(x)
            x = self.encoder.visual.layer3(x) # Remove last layer to have 1024 channels 1024x14x14
            #x = self.encoder.visual.layer4(x)
            
            return x
    
    @property 
    def output_size(self):
        return self._output_size
    

class CLIP_ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, _ = clip.load("ViT-B/16")

        self._output_size = 768

    def forward(self, x, text=False):
        if text:
            x = x.squeeze(1)
            x = self.encoder.encode_text(x)
            x = F.pad(x, (0, 256))
            return x
        else:
            # Ensure the input tensor is of the correct dtype
            x = x.type(self.encoder.visual.conv1.weight.dtype)
            
            # Access the visual transformer components
            visual = self.encoder.visual
            
            # Step 1: Patch Embedding
            x = visual.conv1(x)  # [batch_size, width, grid, grid]
            x = x = rearrange(x, 'b c h w -> b c (h w)')  # [batch_size, width, grid**2]
            x = x.permute(0, 2, 1)  # [batch_size, grid**2, width]
            
            # Step 2: Add Class Token
            x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # [batch_size, 1 + grid**2, width]
            
            # Step 3: Add Positional Embedding
            x = x + visual.positional_embedding.to(x.dtype)  # [batch_size, 1 + grid**2, width]
            
            # Step 4: Apply Layer Norm before Transformer
            x = visual.ln_pre(x)  # [batch_size, 1 + grid**2, width]
            
            # Step 5: Pass through Transformer
            x = rearrange(x, 'b M d -> M b d') # [sequence length, batch size, embedding dimension]
            x = visual.transformer(x)
            x = rearrange(x, 'M b d -> b M d') # [batch size, sequence length, embedding dimension]
            
            # Return all token embeddings
            return x
    
    @property 
    def output_size(self):
        return self._output_size
    

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, in_channels, config=(0, 0, 0, 0, 0, 0), T=12):
        super().__init__()
        self.config = config
        self.shared = nn.ModuleList()
        self.task_specific = nn.ModuleList()
        self.T = T

        # Config 0: Weight
        if self.config[0] == 0:
            self.gamma = nn.Linear(in_features, in_channels, bias=False)
            self.shared.append(self.gamma)
        else:
            self.gamma = nn.ModuleList(
                [nn.Linear(in_features, in_channels, bias=False) for _ in range(self.T)]
            )
            self.task_specific.append(self.gamma)

        # Config 1: Bias
        if self.config[1] == 0:
            self.beta = nn.Linear(in_features, in_channels, bias=False)
            self.shared.append(self.beta)
        else:
            self.beta = nn.ModuleList(
                [nn.Linear(in_features, in_channels, bias=False) for _ in range(self.T)]
            )
            self.task_specific.append(self.beta)

        # Config 2: Convolutional layer I
        if self.config[2] == 0:
            depthwise1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels
            )
            pointwise1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 8,
                kernel_size=1
            )
            self.conv1 = nn.Sequential(depthwise1, pointwise1)
            self.shared.append(self.conv1)
        else:
            self.conv1 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            groups=in_channels
                        ),
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels // 8,
                            kernel_size=1
                        )
                    )
                    for _ in range(self.T)
                ]
            )
            self.task_specific.append(self.conv1)
            
        # Config 3: Convolutional layer II
        if self.config[3] == 0:
            depthwise2 = nn.Conv2d(
                in_channels=in_channels // 8,
                out_channels=in_channels // 8,
                kernel_size=3,
                padding=1,
                groups=in_channels // 8
            )
            pointwise2 = nn.Conv2d(
                in_channels=in_channels // 8,
                out_channels=in_channels,
                kernel_size=1
            )
            self.conv2 = nn.Sequential(depthwise2, pointwise2)
            self.shared.append(self.conv2)
        else:
            self.conv2 = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels // 8,
                            out_channels=in_channels // 8,
                            kernel_size=3,
                            padding=1,
                            groups=in_channels // 8
                        ),
                        nn.Conv2d(
                            in_channels=in_channels // 8,
                            out_channels=in_channels,
                            kernel_size=1
                        )
                    )
                    for _ in range(self.T)
                ]
            )
            self.task_specific.append(self.conv2)

        # Config 4: Batch norm I
        if self.config[4] == 0:
            self.bn1 = nn.BatchNorm2d(in_channels // 8)
            self.shared.append(self.bn1)
        else:
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(in_channels // 8) for _ in range(self.T)])
            self.task_specific.append(self.bn1)

        # Config 5: Batch norm II
        if self.config[5] == 0:
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.shared.append(self.bn2)
        else:
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(self.T)])
            self.task_specific.append(self.bn2)

    def forward(self, x, y, t):
        # Convolutional layer I
        if self.config[2] == 0:
            x = self.conv1(x)
        else:
            if isinstance(t, int):
                x = self.conv1[t](x)
            else:
                x_ = []
                for i in range(len(t)):
                    x_.append(self.conv1[t[i]](x[i]))
                x = torch.stack(x_)

        # Batch norm I
        if self.config[4] == 0:
            x = self.bn1(x)
        else:
            if isinstance(t, int):
                x = self.bn1[t](x)
            else:
                x_ = []
                for i in range(len(t)):
                    x_.append(self.bn1[t[i]](x[i].unsqueeze(0)))
                x = torch.cat(x_)
            
        # RelU
        x.relu_()

        # Convolutional layer II
        if self.config[3] == 0:
            x = self.conv2(x)
        else:
            if isinstance(t, int):
                x = self.conv2[t](x)
            else:
                x_ = []
                for i in range(len(t)):
                    x_.append(self.conv2[t[i]](x[i]))
                x = torch.stack(x_)

        # Weight
        if self.config[0] == 0:
            w = self.gamma(y)
        else:
            if isinstance(t, int):
                w = self.gamma[t](y)
            else:
                w_ = []
                for i in range(len(t)):
                    w_.append(self.gamma[t[i]](y[i]))
                w = torch.stack(w_)

        # Bias
        if self.config[1] == 0:
            b = self.beta(y)
        else:
            if isinstance(t, int):
                b = self.beta[t](y)
            else:
                b_ = []
                for i in range(len(t)):
                    b_.append(self.beta[t[i]](y[i]))
                b = torch.stack(b_)

        # Linear modulation
        x = x * w.unsqueeze(-1).unsqueeze(-1) + b.unsqueeze(-1).unsqueeze(-1)

        # Batch norm II
        if self.config[5] == 0:
            x = self.bn2(x)
        else:
            if isinstance(t, int):
                x = self.bn2[t](x)
            else:
                x_ = []
                for i in range(len(t)):
                    x_.append(self.bn2[t[i]](x[i].unsqueeze(0)))
                x = torch.cat(x_)

        # RelU
        x.relu_()

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=64,
        dropout=0.1,
        batch_first=True,
        config=(0, 0, 0, 0, 0),
        T=12
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.config = config
        self.shared = nn.ModuleList()
        self.task_specific = nn.ModuleList()
        self.T = T

        # Config 0: Multihead attention (Head 1)
        if self.config[0] == 0:
            self.head1 = nn.MultiheadAttention(
                d_model, 1, dropout=dropout, batch_first=batch_first
            )
            self.shared.append(self.head1)
        else:
            self.head1 = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model, 1, dropout=dropout, batch_first=batch_first
                    )
                    for _ in range(self.T)
                ]
            )
            self.task_specific.append(self.head1)

        # Config 1: Multihead attention (Head 2)
        if self.config[1] == 0:
            self.head2 = nn.MultiheadAttention(
                d_model, 1, dropout=dropout, batch_first=batch_first
            )
            self.shared.append(self.head2)
        else:
            self.head2 = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model, 1, dropout=dropout, batch_first=batch_first
                    )
                    for _ in range(self.T)
                ]
            )
            self.task_specific.append(self.head2)

        # Config 2: Feed-forward layer I
        if self.config[2] == 0:
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.shared.append(self.ffn1)
        else:
            self.ffn1 = nn.ModuleList(
                [nn.Linear(d_model, dim_feedforward) for _ in range(self.T)]
            )
            self.task_specific.append(self.ffn1)

        # Config 3: Feed-forward layer II
        if self.config[3] == 0:
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
            self.shared.append(self.ffn2)
        else:
            self.ffn2 = nn.ModuleList(
                [nn.Linear(dim_feedforward, d_model) for _ in range(self.T)]
            )
            self.task_specific.append(self.ffn2)

        # Config 4: Layer normalization I
        if self.config[4] == 0:
            self.ln2 = nn.LayerNorm(d_model)
            self.shared.append(self.ln2)
        else:
            self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.T)])
            self.task_specific.append(self.ln2)

        # Config 5: Layer normalization II
        if self.config[5] == 0:
            self.ln1 = nn.LayerNorm(d_model)
            self.shared.append(self.ln1)
        else:
            self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.T)])
            self.task_specific.append(self.ln1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # Layer normalization I
        if self.config[4] == 0:
            x_ = self.ln2(x)
        else:
            if isinstance(t, int):
                x_ = self.ln2[t](x)
            else:
                x__ = []
                for i in range(len(t)):
                    x__.append(self.ln2[t[i]](x[i]))
                x_ = torch.stack(x__)

        # Multihead attention (Head 1)
        if self.config[0] == 0:
            x1, _ = self.head1(x_, x_, x_)
        else:
            if isinstance(t, int):
                x1, _ = self.head1[t](x_, x_, x_)
            else:
                x1__ = []
                for i in range(len(t)):
                    x1__.append(self.head1[t[i]](x_[i], x_[i], x_[i])[0])
                x1 = torch.stack(x1__)

        # Multihead attention (Head 2)
        if self.config[1] == 0:
            x2, _ = self.head2(x_, x_, x_)
        else:
            if isinstance(t, int):
                x2, _ = self.head2[t](x_, x_, x_)
            else:
                x2__ = []
                for i in range(len(t)):
                    x2__.append(self.head2[t[i]](x_[i], x_[i], x_[i])[0])
                x2 = torch.stack(x2__)

        x = x + self.dropout(x1 + x2)

        # Layer normalization II
        if self.config[5] == 0:
            x_ = self.ln1(x)
        else:
            if isinstance(t, int):
                x_ = self.ln1[t](x)
            else:
                x__ = []
                for i in range(len(t)):
                    x__.append(self.ln1[t[i]](x[i]))
                x_ = torch.stack(x__)

        # Feed-forward layer I
        if self.config[2] == 0:
            x_ = self.ffn1(x_)
        else:
            if isinstance(t, int):
                x_ = self.ffn1[t](x_)
            else:
                x__ = []
                for i in range(len(t)):
                    x__.append(self.ffn1[t[i]](x_[i]))
                x_ = torch.stack(x__)

        # ReLU
        x_.relu_()

        # Feed-forward layer II
        if self.config[3] == 0:
            x_ = self.ffn2(x_)
        else:
            if isinstance(t, int):
                x_ = self.ffn2[t](x_)
            else:
                x__ = []
                for i in range(len(t)):
                    x__.append(self.ffn2[t[i]](x_[i]))
                x_ = torch.stack(x__)

        out = x + self.dropout(x_)

        return out
