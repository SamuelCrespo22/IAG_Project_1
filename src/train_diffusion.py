import torch
from torch.optim import Adam
from models.diffusion import UNet32, DDPM
from tqdm import tqdm
import time
# Import dataloader from here: from dataset import get_dataloader

def train_diffusion(dataloader, num_epochs=50, device="cuda", lr=2e-4, T=1000, time_dim=128):
    print("[INFO] Initializing U-Net and DDPM...")
    
    # Initialize network and diffusion process
    unet = UNet32(time_dim=time_dim).to(device)
    
    # T=1000 is the standard for high quality.
    # For quick testing, your colleague can change to T=250.
    ddpm = DDPM(model=unet, T=T, device=device) 
    
    optimizer = Adam(unet.parameters(), lr=lr)

    unet.train()
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch_idx, (images, _) in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # All the magic happens here: the DDPM class chooses 't',
            # adds noise, and calculates MSE Loss against U-Net prediction
            loss = ddpm.loss(images)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss Diffusion (MSE): {avg_loss:.4f} | Time: {int(epoch_mins)}m {epoch_secs:.0f}s")

    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)
    print(f"[INFO] Diffusion training completed in {int(total_mins)}m {total_secs:.0f}s.")
    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0
    return unet, ddpm, total_duration, avg_epoch_duration