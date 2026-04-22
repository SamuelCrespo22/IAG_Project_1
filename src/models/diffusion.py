import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================================================
# Sinusoidal Time Embedding
# ======================================================
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


# ======================================================
# Convolutional Block with GroupNorm + Time Embedding
# ======================================================
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


# ======================================================
# Self-Attention Block (Spatial Attention)
# ======================================================
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # O GroupNorm ajuda a estabilizar o treino, igual às convoluções
        self.group_norm = nn.GroupNorm(8, channels)
        
        # Convoluções 1x1 para gerar as Queries, Keys e Values (Q, K, V)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        
        # Convolução 1x1 final para projetar o resultado
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Normalizar a entrada
        h = self.group_norm(x)
        
        # 2. Gerar Q, K e V
        qkv = self.qkv(h) # Fica com tamanho (B, 3*C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1) # Separa em 3 tensores de tamanho (B, C, H, W)
        
        # 3. Redimensionar (Flatten) a altura e largura (H*W) para calcular a atenção
        # Transforma de (B, C, H, W) para (B, C, N) onde N = H*W
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        # 4. Calcular os "Attention Scores" (Q * K^T / sqrt(C))
        # q.transpose passa a (B, N, C). Multiplica por k (B, C, N) -> resultado (B, N, N)
        attn_scores = torch.bmm(q.transpose(1, 2), k) * (C ** (-0.5))
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 5. Aplicar os scores aos Values (V)
        # attn_probs (B, N, N) * v.transpose (B, N, C) -> (B, N, C)
        out = torch.bmm(attn_probs, v.transpose(1, 2))
        
        # 6. Voltar ao formato de imagem original (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # 7. Projeção final e "Residual Connection" (soma com a entrada original)
        out = self.proj(out)
        return x + out


# ======================================================
# U-Net 32x32 with Self-Attention in Bottleneck
# ======================================================
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
        self.mid1 = ConvBlock(256, 256, time_dim)
        self.mid_attn = SelfAttention(256)
        self.mid2 = ConvBlock(256, 256, time_dim)

        # -------- Decoder --------
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 128, time_dim)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 64, time_dim)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64, time_dim)

        # Output: predicted noise
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # -------- Encoder --------
        x1 = self.enc1(x, t_emb)              # 32x32, 64
        x2 = self.enc2(self.pool(x1), t_emb)  # 16x16, 128
        x3 = self.enc3(self.pool(x2), t_emb)  #  8x8, 256

        # -------- Bottleneck --------
        h = self.mid1(self.pool(x3), t_emb)   #  4x4, 256
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # -------- Decoder --------
        h = self.up3(h)
        h = self.dec3(torch.cat([h, x3], dim=1), t_emb)

        h = self.up2(h)
        h = self.dec2(torch.cat([h, x2], dim=1), t_emb)

        h = self.up1(h)
        h = self.dec1(torch.cat([h, x1], dim=1), t_emb)

        return self.out(h)

# ======================================================
# Cosine Beta Schedule (Nichol & Dhariwal, 2021)
# ======================================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5
    ) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.999)


# ======================================================
# DDPM
# ======================================================
class DDPM:
    def __init__(self, model, T=1000, device="cpu"):
        self.model = model.to(device)
        self.T = T
        self.device = device

        self.betas = cosine_beta_schedule(T).to(device)
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