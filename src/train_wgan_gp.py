import torch
from torch import nn
from torch.optim import Adam
from models.wgan_gp import Generator, Critic, weights_init
from tqdm import tqdm
import time

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # Mix real and fake samples.
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    fake_labels = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calculate the gradient norm and penalize if it deviates from 1.
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(dataloader, num_epochs=100, device="cuda", lr=1e-4, latent_dim=100):
    gen = Generator(inputDim=latent_dim).to(device)
    gen.apply(weights_init)
    
    critic = Critic().to(device)
    critic.apply(weights_init)

    genOpt = Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    criticOpt = Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    lambda_gp = 10 # Weight of the Gradient Penalty.
    n_critic = 5   # Train the Critic 5 times for each time the Generator is trained.

    print("[INFO] Starting WGAN-GP training...")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for i, (images, _) in progress_bar:
            real_imgs = images.to(device)
            bs = real_imgs.size(0)

            # ==========================================
            # TRAIN THE CRITIC (n_critic times)
            # ==========================================
            criticOpt.zero_grad()

            real_validity = critic(real_imgs)
            
            z = torch.randn(bs, latent_dim, 1, 1, device=device)
            fake_imgs = gen(z)
            fake_validity = critic(fake_imgs.detach()) # Detach so we don't train the Generator here

            gradient_penalty = compute_gradient_penalty(critic, real_imgs.detach(), fake_imgs.detach(), device)

            # Critic Loss: (Fake mean - Real mean) + penalty
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            criticOpt.step()

            # ==========================================
            # TRAIN THE GENERATOR (Only every n_critic times)
            # ==========================================
            if i % n_critic == 0:
                genOpt.zero_grad()

                z = torch.randn(bs, latent_dim, 1, 1, device=device)
                gen_imgs = gen(z)
                fake_validity = critic(gen_imgs)

                # Try to maximize the Critic's score (minimize the negative)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                genOpt.step()

            progress_bar.set_postfix({"Loss Critic": f"{d_loss.item():.4f}", "Loss Gen": f"{g_loss.item():.4f}"})

        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss Critic: {d_loss.item():.4f} | Loss Gen: {g_loss.item():.4f} | Time: {int(epoch_mins)}m {epoch_secs:.0f}s")
        
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)
    print(f"[INFO] WGAN-GP training completed in {int(total_mins)}m {total_secs:.0f}s.")
    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0
    return gen, total_duration, avg_epoch_duration