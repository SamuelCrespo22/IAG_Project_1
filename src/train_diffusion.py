import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

from models.diffusion import UNet32, DDPM


def train_diffusion(
    dataloader,
    num_epochs=500,
    device="cuda",
    lr=1e-4,
    timesteps=500
):
    # ======================================================
    # Model + Diffusion process
    # ======================================================
    unet = UNet32().to(device)
    ddpm = DDPM(unet, T=timesteps, device=device)

    optimizer = Adam(unet.parameters(), lr=lr)

    scheduler = StepLR(
        optimizer,
        step_size=num_epochs // 2,  # reduzir LR a meio do treino
        gamma=0.5                   # LR = LR * 0.5
    )

    total_start_time = time.time()

    # ======================================================
    # Training loop
    # ======================================================
    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=False
        )

        for images, _ in progress_bar:
            images = images.to(device)

            loss = ddpm.loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.6f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        # --------------------------------------------------
        # Epoch stats
        # --------------------------------------------------
        epoch_duration = time.time() - epoch_start_time
        mins, secs = divmod(epoch_duration, 60)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Loss: {epoch_loss / len(dataloader):.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {int(mins)}m {secs:.0f}s"
        )

    # ======================================================
    # Total time
    # ======================================================
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)

    print(
        f"[INFO] Diffusion training completed in "
        f"{int(total_mins)}m {total_secs:.0f}s."
    )

    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0

    return unet, ddpm, total_duration, avg_epoch_duration


# def train_diffusion(
#     dataloader,
#     num_epochs=50,
#     device="cuda",
#     lr=2e-4,
#     timesteps=1000
# ):
#     unet = UNet32().to(device)
#     ddpm = DDPM(unet, T=timesteps, device=device)

#     optimizer = Adam(unet.parameters(), lr=lr)

#     total_start_time = time.time()

#     for epoch in range(num_epochs):
#         unet.train()
#         epoch_loss = 0
#         epoch_start_time = time.time()

#         progress_bar = tqdm(
#             enumerate(dataloader),
#             total=len(dataloader),
#             desc=f"Epoch [{epoch+1}/{num_epochs}]",
#             leave=False
#         )

#         for i, (images, _) in progress_bar:
#             images = images.to(device) * 2 - 1

#             loss = ddpm.loss(images)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#             progress_bar.set_postfix({
#                 "Loss": f"{loss.item():.6f}"
#             })

#         epoch_duration = time.time() - epoch_start_time
#         epoch_mins, epoch_secs = divmod(epoch_duration, 60)

#         print(
#             f"Epoch [{epoch+1}/{num_epochs}] "
#             f"Loss: {epoch_loss/len(dataloader):.6f} "
#             f"Time: {int(epoch_mins)}m {epoch_secs:.0f}s"
#         )

#     total_duration = time.time() - total_start_time
#     total_mins, total_secs = divmod(total_duration, 60)

#     print(
#         f"[INFO] Diffusion training completed in "
#         f"{int(total_mins)}m {total_secs:.0f}s."
#     )

#     avg_epoch_duration = (
#         total_duration / num_epochs if num_epochs > 0 else 0
#     )

#     return unet, ddpm, total_duration, avg_epoch_duration


# def train_diffusion(
#     dataloader,
#     num_epochs=50,
#     device="cuda",
#     lr=2e-4,
#     timesteps=1000
# ):
#     unet = UNet2DModel(
#         sample_size=32,          # Resolução da imagem alvo
#         in_channels=3,           # Canais de entrada (RGB)
#         out_channels=3,          # Canais de saída
#         layers_per_block=2,      # Camadas ResNet por bloco
#         block_out_channels=(128, 256, 256, 512), 
#         down_block_types=(
#             "DownBlock2D",       # 32x32 -> 16x16
#             "DownBlock2D",       # 16x16 -> 8x8
#             "AttnDownBlock2D",   # 8x8 -> 4x4 (Atenção espacial)
#             "DownBlock2D",       # 4x4 -> 2x2
#         ),
#         up_block_types=(
#             "UpBlock2D",         # 2x2 -> 4x4
#             "AttnUpBlock2D",     # 4x4 -> 8x8 (Atenção espacial)
#             "UpBlock2D",         # 8x8 -> 16x16
#             "UpBlock2D",         # 16x16 -> 32x32
#         ),
#     ).to(device)

#     ddpm = DDPM(unet, T=timesteps, device=device)

#     optimizer = Adam(unet.parameters(), lr=lr)

#     total_start_time = time.time()

#     for epoch in range(num_epochs):
#         unet.train()
#         epoch_loss = 0
#         epoch_start_time = time.time()

#         progress_bar = tqdm(
#             enumerate(dataloader),
#             total=len(dataloader),
#             desc=f"Epoch [{epoch+1}/{num_epochs}]",
#             leave=False
#         )

#         for i, (images, _) in progress_bar:
#             # Normalização para [-1, 1]
#             images = images.to(device) * 2 - 1

#             loss = ddpm.loss(images)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#             progress_bar.set_postfix({
#                 "Loss": f"{loss.item():.6f}"
#             })

#         epoch_duration = time.time() - epoch_start_time
#         epoch_mins, epoch_secs = divmod(epoch_duration, 60)

#         print(
#             f"Epoch [{epoch+1}/{num_epochs}] "
#             f"Loss: {epoch_loss/len(dataloader):.6f} "
#             f"Time: {int(epoch_mins)}m {epoch_secs:.0f}s"
#         )

#     total_duration = time.time() - total_start_time
#     total_mins, total_secs = divmod(total_duration, 60)

#     print(
#         f"[INFO] Diffusion training completed in "
#         f"{int(total_mins)}m {total_secs:.0f}s."
#     )

#     avg_epoch_duration = (
#         total_duration / num_epochs if num_epochs > 0 else 0
#     )

#     return unet, ddpm, total_duration, avg_epoch_duration