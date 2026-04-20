import torch
from torch import nn
from torch.optim import Adam
from models.dcgan import Generator, Discriminator, weights_init
from tqdm import tqdm
import time

def train_gan(dataloader, num_epochs=20, device="cuda", lr=0.0002, latent_dim=100):
    gen = Generator(inputDim=latent_dim).to(device)
    gen.apply(weights_init)

    disc = Discriminator().to(device)
    disc.apply(weights_init)

    genOpt = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    discOpt = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    realLabel, fakeLabel = 1.0, 0.0

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epochLossG, epochLossD = 0, 0
        epoch_start_time = time.time()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for i, (images, _) in progress_bar:
            images = images.to(device)
            bs = images.size(0)

            discOpt.zero_grad()
            labels = torch.full((bs,), realLabel, dtype=torch.float, device=device)
            output = disc(images).view(-1)
            errorReal = criterion(output, labels)
            errorReal.backward()

            noise = torch.randn(bs, latent_dim, 1, 1, device=device)
            fake = gen(noise)
            labels.fill_(fakeLabel)
            output = disc(fake.detach()).view(-1)
            errorFake = criterion(output, labels)
            errorFake.backward()

            errorD = errorReal + errorFake
            discOpt.step()

            genOpt.zero_grad()
            labels.fill_(realLabel) # To train the generator, we want fake images to pass as real
            output = disc(fake).view(-1)
            errorG = criterion(output, labels)
            errorG.backward()
            genOpt.step()

            epochLossD += errorD.item()
            epochLossG += errorG.item()
            
            progress_bar.set_postfix({"Loss D": f"{errorD.item():.4f}", "Loss G": f"{errorG.item():.4f}"})

        epoch_duration = time.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(epoch_duration, 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {epochLossD/len(dataloader):.4f} | Loss G: {epochLossG/len(dataloader):.4f} | Time: {int(epoch_mins)}m {epoch_secs:.0f}s")
        
    total_duration = time.time() - total_start_time
    total_mins, total_secs = divmod(total_duration, 60)
    print(f"[INFO] DCGAN training completed in {int(total_mins)}m {total_secs:.0f}s.")
    avg_epoch_duration = total_duration / num_epochs if num_epochs > 0 else 0
    return gen, total_duration, avg_epoch_duration