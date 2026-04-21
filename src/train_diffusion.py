import torch
from torch.optim import Adam
from tqdm import tqdm
import time

from models.diffusion import UNet32, DDPM


def train_diffusion(
    dataloader,
    num_epochs=50,
    device="cuda",
    lr=2e-4,
    timesteps=1000
):
    # ----------------------------------------
    # Modelo e otimizador
    # ----------------------------------------
    unet = UNet32().to(device)
    ddpm = DDPM(unet, T=timesteps, device=device)

    optimizer = Adam(unet.parameters(), lr=lr)

    total_start_time = time.time()

    # ----------------------------------------
    # Loop de treino
    # ----------------------------------------
    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            leave=False
        )

        for i, (images, _) in progress_bar:
            # Normalização para [-1, 1]
            images = images.to(device) * 2 - 1

            loss = ddpm.loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.6f}"
            })

        # ------------------------------------
        # Estatísticas do epoch
        # ------------------------------------
        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {epoch_loss/len(dataloader):.6f} "
            f"Time: {int(epoch_mins)}m {epoch_secs:.0f}s"
        )

    # ----------------------------------------
    # Tempo total
    # ----------------------------------------
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)

    print(
        f"[INFO] Diffusion training completed in "
        f"{int(total_mins)}m {total_secs:.0f}s."
    )

    avg_epoch_duration = (
        total_duration / num_epochs if num_epochs > 0 else 0
    )

    return unet, ddpm, total_duration, avg_epoch_duration
