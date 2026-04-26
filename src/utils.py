import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def save_model(model, filename):
    """Saves the model weights."""
    torch.save(model.state_dict(), filename)
    print(f"[INFO] Model successfully saved to {filename}")

def generate_and_save_visual_grid(model, device, latent_dim, model_type="vae", filename="visual_grid.png"):
    """
    Generates 64 random images and saves them in an 8x8 grid.
    """
    model.eval()
    with torch.no_grad():
        # Convolutional GANs need the format (batch, channels, height, width).
        if model_type in ["gan", "wgan"]:
            z = torch.randn(64, latent_dim, 1, 1).to(device)
        else:
            z = torch.randn(64, latent_dim).to(device)
        
        if model_type == "vae":
            fake_images = model.decode(z)
        else:
            fake_images = model(z)
            
        # Denormalize for Matplotlib visualization.
        if model_type in ["gan", "wgan", "diffusion"]:
            fake_images = (fake_images + 1.0) / 2.0
            
        fake_images = torch.clamp(fake_images, 0.0, 1.0)
        grid = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=False)
        
        grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Samples ({model_type.upper()})")
        plt.imshow(grid_np)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Visual grid saved to {filename}")

def log_experiment_to_csv(csv_file, exp_name, model_type, epochs, batch_size, lr, latent_dim, extra_notes, all_fid, all_kid, all_is, all_lpips, total_time=None, avg_epoch_time=None):
    """
    Logs the hyperparameters and final evaluation metrics to a CSV file.
    Appends a new row if the file exists, creates it with headers if it doesn't.
    """
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['Experiment_Name', 'Model', 'Epochs', 'Batch_Size', 'LR', 'Latent_Dim', 'Notes', 
                             'FID_mean', 'FID_std', 'KID_mean', 'KID_std', 'IS_mean', 'IS_std', 'LPIPS_mean', 'LPIPS_std',
                             'Total_Time_s', 'Avg_Epoch_Time_s'])
        
        t_time_str = f"{total_time:.2f}" if total_time is not None else "N/A"
        avg_ep_time_str = f"{avg_epoch_time:.2f}" if avg_epoch_time is not None else "N/A"
        
        writer.writerow([
            exp_name, model_type, epochs, batch_size, lr, latent_dim, extra_notes,
            f"{np.mean(all_fid):.2f}", f"{np.std(all_fid):.2f}", f"{np.mean(all_kid):.4f}", f"{np.std(all_kid):.4f}",
            f"{np.mean(all_is):.2f}", f"{np.std(all_is):.2f}", f"{np.mean(all_lpips):.4f}", f"{np.std(all_lpips):.4f}",
            t_time_str, avg_ep_time_str
        ])
    print(f"[INFO] Results successfully logged to {csv_file}")