import torch
from torch.optim import Adam
from tqdm import tqdm
import time

from models.vae import ConvVAE, vae_loss


def train_vae(
    dataloader,
    num_epochs=20,
    device="cuda",
    lr=1e-3,
    latent_dim=128
):
    # ----------------------------------------
    # Model and optimizer
    # ----------------------------------------
    vae = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=lr)

    total_start_time = time.time()

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    for epoch in range(num_epochs):
        vae.train()
        epoch_loss = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            leave=False
        )

        for _, (images, _) in progress_bar:
            images = images.to(device)

            recon, mu, logvar = vae(images)
            loss = vae_loss(recon, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ------------------------------------
        # Epoch statistics
        # ------------------------------------
        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {epoch_loss/len(dataloader):.4f} "
            f"Time: {int(epoch_mins)}m {epoch_secs:.0f}s"
        )

    # ----------------------------------------
    # Total time
    # ----------------------------------------
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)

    print(
        f"[INFO] VAE training completed in "
        f"{int(total_mins)}m {total_secs:.0f}s."
    )

    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0

    vae.eval()

    return vae, total_duration, avg_epoch_duration