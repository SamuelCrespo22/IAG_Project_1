import torch
from dataset import get_dataloader
from train_wgan_gp import train_wgan_gp
from models.wgan_gp import Generator
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
    
MODEL_TYPE = "wgan"
USE_SUBSET = True

# --- EXPERIMENT HYPERPARAMETERS ---
EXP_NAME = "WGAN_Exp5_Latent100_BS128"
EPOCHS = 100 # 100 WGAN epochs ~= 20 DCGAN epochs
BATCH_SIZE = 128
LR = 1e-4
LATENT_DIM = 100

print(f"=== STARTING WGAN-GP EXPERIMENT ON DEVICE: {DEVICE} ===")

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
    use_subset=False,
    split="test",
    data_source="kaggle"
)

# =================================================================
# Train WGAN-GP
# =================================================================
generator_wgan, total_time, avg_epoch_time = train_wgan_gp(
    train_loader,
    num_epochs=EPOCHS,
    device=DEVICE,
    lr=LR,
    latent_dim=LATENT_DIM
)

# =================================================================
# Save Model and create Visual Grid
# =================================================================
save_model(generator_wgan, f"weights_{EXP_NAME}.pth")
generate_and_save_visual_grid(
    model=generator_wgan, 
    device=DEVICE, 
    latent_dim=LATENT_DIM, 
    model_type=MODEL_TYPE, 
    filename=f"grid_{EXP_NAME}.png"
)

# =================================================================
# Evaluation
# =================================================================
class WGANGeneratorWrapper(torch.nn.Module):
    def __init__(self, wgan_model, latent_dim=100):
        super().__init__()
        self.wgan_model = wgan_model
        self.latent_dim = latent_dim
        
    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(DEVICE)
        return self.wgan_model(z)

wgan_for_evaluation = WGANGeneratorWrapper(generator_wgan, latent_dim=LATENT_DIM)

print("\n=== STARTING WGAN-GP EVALUATION ===")
all_fid, all_kid, all_is, all_lpips = run_full_evaluation(
    model_type=MODEL_TYPE,
    generator=wgan_for_evaluation,
    real_dataloader=test_loader,
    device=DEVICE
)

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
    extra_notes="Standard WGAN-GP Phase 1",
    all_fid=all_fid, all_kid=all_kid, all_is=all_is, all_lpips=all_lpips,
    total_time=total_time, avg_epoch_time=avg_epoch_time
)

print("=== WGAN-GP EXPERIMENT FULLY COMPLETED! ===")
