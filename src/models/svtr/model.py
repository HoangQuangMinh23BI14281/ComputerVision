import torch
import torch.nn as nn

class ConvMixer(nn.Module):
    def __init__(self, dim):
        super(ConvMixer, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.act(self.norm(h + x))

class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads, mixer_type="Global", drop=0.1):
        super(SVTRBlock, self).__init__()
        self.mixer_type = mixer_type
        if mixer_type == "Global":
            self.mixer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop)
            self.norm1 = nn.LayerNorm(dim)
        else:
            self.mixer = ConvMixer(dim)
            self.norm1 = nn.BatchNorm2d(dim)
            
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape if Global: [B, Seq_len, Dim]
        # x shape if Local: [B, Dim, H, W]
        if self.mixer_type == "Global":
            h = self.norm1(x)
            h, _ = self.mixer(h, h, h)
            x = x + h
            # MLP
            x = x + self.mlp(self.norm2(x))
        else:
            # ConvMixer operates on spatial dimensions
            # Apply Pre-Norm
            h = self.norm1(x)
            h = self.mixer(h)
            x = x + h
            # Apply MLP: requires reshaping to Sequence
            b, c, h_idx, w_idx = x.shape
            x_flat = x.flatten(2).transpose(1, 2) # [B, H*W, C]
            x_flat = x_flat + self.mlp(self.norm2(x_flat))
            x = x_flat.transpose(1, 2).reshape(b, c, h_idx, w_idx)
        return x
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, Seq_len, Dim]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

import math

class SVTRv2(nn.Module):
    def __init__(self, imgH=32, nc=3, nclass=37, embed_dims=[64, 128, 256], depths=[3, 6, 3], num_heads=[2, 4, 8], drop=0.1):
        super(SVTRv2, self).__init__()
        # Simplified SVTR architecture
        self.patch_embed = nn.Sequential(
            nn.Conv2d(nc, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Dropout2d(drop)
        )
        
        # Stage 1: Local Mixing (Conv)
        self.stage1 = nn.Sequential(*[SVTRBlock(embed_dims[0], num_heads[0], "Local", drop=drop) for _ in range(depths[0])])
        self.down1 = nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=(2, 1), padding=1)
        
        # Positional Encoding for Stage 2 & 3
        self.pos_embed2 = PositionalEncoding(embed_dims[1])
        self.pos_embed3 = PositionalEncoding(embed_dims[2])

        # Stage 2: Global Mixing (Transformer)
        self.stage2 = nn.Sequential(*[SVTRBlock(embed_dims[1], num_heads[1], "Global", drop=drop) for _ in range(depths[1])])
        self.down2 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=(2, 1), padding=1)
        
        # Stage 3: Global Mixing (Transformer)
        self.stage3 = nn.Sequential(*[SVTRBlock(embed_dims[2], num_heads[2], "Global", drop=drop) for _ in range(depths[2])])
        
        self.head = nn.Linear(embed_dims[2], nclass)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x) # [B, D0, H/4, W/4]
        
        x = self.stage1(x) # Local mixing
        x = self.down1(x) # [B, D1, H/8, W/4]
        
        # Convert to sequence for Global Attention
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, H*W, c]
        x_flat = self.pos_embed2(x_flat) # Add Positional Encoding
        x_flat = self.stage2(x_flat)
        x = x_flat.transpose(1, 2).reshape(b, c, h, w)
        
        x = self.down2(x) # [B, D2, H/16, W/4]
        
        # Force height down to 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, x.shape[3]))
        
        x_flat = x.squeeze(2).transpose(1, 2) # [B, W, C]
        x_flat = self.pos_embed3(x_flat) # Add Positional Encoding
        x_flat = self.stage3(x_flat) # [B, seq_len, C]
        
        x_out = self.head(x_flat) # [B, seq_len, NClass]
        
        # CTC loss component expects [seq_len, batch_size, num_classes]
        return x_out.transpose(0, 1)

if __name__ == '__main__':
    model = SVTRv2(imgH=32, nc=3, nclass=37)
    dummy_input = torch.randn(2, 3, 32, 320) # batch_size=2, width=320
    out = model(dummy_input)
    print("SVTRv2 output shape:", out.shape) # Expected: [seq_len, batch_size, nclass]
