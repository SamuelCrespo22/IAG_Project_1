import torch
from dataset import get_dataloader
from train_vae import train_vae
from evaluate import run_full_evaluation
from utils import save_model, generate_and_save_visual_grid, log_experiment_to_csv

# =================================================================
# Configurations
# =================================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" 
else:
    DEVICE = "cpu"

MODEL_TYPE = "vae"
USE_SUBSET = True  # True for Phase 1 (20%). False for Phase 2 (100%).

# --- EXPERIMENT HYPERPARAMETERS ---
EXP_NAME = "VAE_Exp5_Latent256_Lr5e-4"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
LATENT_DIM = 256

print(f"=== STARTING VAE EXPERIMENT ON DEVICE: {DEVICE} ===")

# =================================================================
# Load Data
# =================================================================
train_loader = get_dataloader(
    model_type=MODEL_TYPE, 
    batch_size=BATCH_SIZE, 
    use_subset=USE_SUBSET,
    split="train",
    data_source="kaggle"
)

test_loader = get_dataloader(
    model_type=MODEL_TYPE, 
    batch_size=BATCH_SIZE, 
    use_subset=False, # The test split must have enough images for the 5000 samples.
    split="test",
    data_source="kaggle"
)

# =================================================================
# Train the VAE
# =================================================================
trained_vae, total_time, avg_epoch_time = train_vae(
    train_loader,
    num_epochs=EPOCHS,
    device=DEVICE,
    lr=LR,
    latent_dim=LATENT_DIM
)

# =================================================================
# Save Model and create Visual Grid
# =================================================================
save_model(trained_vae, f"weights_{EXP_NAME}.pth")
generate_and_save_visual_grid(
    model=trained_vae,
    device=DEVICE,
    latent_dim=LATENT_DIM,
    model_type=MODEL_TYPE,
    filename=f"grid_{EXP_NAME}.png"
)

# =================================================================
# Evaluation
# =================================================================
# For evaluate to accept the VAE, create a "wrapper" function that generates
# the fake images using the .decode() method instead of .forward()
class VAEGeneratorWrapper(torch.nn.Module):
    def __init__(self, vae_model, latent_dim=128):
        super().__init__()
        self.vae_model = vae_model
        self.latent_dim = latent_dim
        
    def forward(self, batch_size):
        # evaluate.py will call this
        z = torch.randn(batch_size, self.latent_dim).to(DEVICE)
        return self.vae_model.decode(z)

vae_for_evaluation = VAEGeneratorWrapper(trained_vae, latent_dim=LATENT_DIM)

print("\n=== STARTING VAE EVALUATION ===")
all_fid, all_kid, all_is, all_lpips = run_full_evaluation(model_type=MODEL_TYPE, generator=vae_for_evaluation, real_dataloader=test_loader, device=DEVICE)

# =================================================================
# Log the results
# =================================================================
log_experiment_to_csv(
    csv_file="phase1_results.csv",
    exp_name=EXP_NAME,
    model_type=MODEL_TYPE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    latent_dim=LATENT_DIM,
    extra_notes="Standard VAE Phase 1",
    all_fid=all_fid, all_kid=all_kid, all_is=all_is, all_lpips=all_lpips,
    total_time=total_time, avg_epoch_time=avg_epoch_time
)

print("=== VAE EXPERIMENT FULLY COMPLETED ===")
