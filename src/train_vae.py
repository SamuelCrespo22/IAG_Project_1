import torch
from torch.optim import Adam
from models.vae import ConvVAE, vae_loss
from tqdm import tqdm
import time

def train_vae(dataloader, num_epochs=50, device="cuda", lr=1e-3, latent_dim=128):
    print("[INFO] Starting VAE training.")
    
    vae = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=lr)
    
    vae.train()
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for i, (images, _) in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = vae(images)
            
            loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss VAE: {avg_loss:.4f} | Time: {int(epoch_mins)}m {epoch_secs:.0f}s")
        
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)
    print(f"[INFO] VAE training completed in {int(total_mins)}m {total_secs:.0f}s.")
    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0
    return vae, total_duration, avg_epoch_duration
