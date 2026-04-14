import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Variational Autoencoder Convolucional
# ======================================================

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # ---------------- Encoder ----------------
        # Recebe imagens (3, 32, 32) e extrai features espaciais
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> (128, 4, 4)
        )

        # Camadas que produzem média e variância do espaço latente
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # ---------------- Decoder ----------------
        # Expande o vetor latente de volta para um mapa de features
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)

        # Reconstrói a imagem usando convoluções transpostas
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # -> (3, 32, 32)
            nn.Sigmoid()  # Garante valores entre 0 e 1
        )

    # --------------------------------------------------
    # Encoder: imagem → parâmetros da distribuição
    # --------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # --------------------------------------------------
    # Reparametrização: amostra diferenciável
    # --------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------------------------------
    # Decoder: espaço latente → imagem
    # --------------------------------------------------
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder(h)

    # --------------------------------------------------
    # Passo completo do VAE
    # --------------------------------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ======================================================
# Função de perda do VAE
# ======================================================
def vae_loss(recon_x, x, mu, logvar):
    """
    Combina:
    - erro de reconstrução da imagem
    - divergência KL (regularização do espaço latente)
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
