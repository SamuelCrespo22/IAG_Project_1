import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =================================================================
# Sinusoidal Time Embedding (Encodes the timestep 't')
# =================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# =================================================================
# Convolutional Block with GroupNorm and Time Embedding injection
# =================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)

        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return F.relu(h + t)


# =================================================================
# Symmetric U-Net for 32x32 images
# =================================================================
class UNet32(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # -------- Encoder --------
        self.enc1 = ConvBlock(3, 64, time_dim)
        self.enc2 = ConvBlock(64, 128, time_dim)
        self.enc3 = ConvBlock(128, 256, time_dim)

        self.pool = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.mid = ConvBlock(256, 256, time_dim)

        # -------- Decoder --------
        # Fix: Ensure channels match exactly after torch.cat
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        # 256 (from up3) + 256 (from x3) = 512 input channels
        self.dec3 = ConvBlock(512, 128, time_dim) 

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        # 128 (from up2) + 128 (from x2) = 256 input channels
        self.dec2 = ConvBlock(256, 64, time_dim)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        # 64 (from up1) + 64 (from x1) = 128 input channels
        self.dec1 = ConvBlock(128, 64, time_dim)

        # Final Output: Predict the noise added to the image
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # -------- Encoder --------
        x1 = self.enc1(x, t_emb)              # Output: 32x32, 64 ch
        x2 = self.enc2(self.pool(x1), t_emb)  # Output: 16x16, 128 ch
        x3 = self.enc3(self.pool(x2), t_emb)  # Output: 8x8,  256 ch

        # -------- Bottleneck --------
        h = self.mid(self.pool(x3), t_emb)    # Output: 4x4,  256 ch

        # -------- Decoder --------
        h = self.up3(h)                       # Output: 8x8, 256 ch
        h = self.dec3(torch.cat([h, x3], dim=1), t_emb)

        h = self.up2(h)                       # Output: 16x16, 128 ch
        h = self.dec2(torch.cat([h, x2], dim=1), t_emb)

        h = self.up1(h)                       # Output: 32x32, 64 ch
        # Notice we concatenate with x1 (Encoder feature map), not x (Raw image)
        h = self.dec1(torch.cat([h, x1], dim=1), t_emb)

        return self.out(h)


# ======================================================
# Scheduler
# ======================================================
def linear_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)


# =================================================================
# DDPM (Denoising Diffusion Probabilistic Models) Process
# =================================================================
class DDPM:
    def __init__(self, model, T=1000, device="cpu"):
        self.model = model.to(device)
        self.T = T
        self.device = device

        self.betas = linear_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        a_bar = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise, noise

    def loss(self, x0):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_noisy, noise = self.forward_diffusion(x0, t)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n):
        x = torch.randn(n, 3, 32, 32, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, t_batch)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.betas[t]

            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * eps
            ) + torch.sqrt(beta) * noise

        return x