import torch
from dataset import get_dataloader
from train_dcgan import train_gan
from models.dcgan import Generator
from evaluate import run_full_evaluation
from utils import save_model, generate_and_save_visual_grid, log_experiment_to_csv

# =================================================================
# Configurations
# =================================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" # Activates the Mac GPU (Apple Silicon M1/M2/M3)
else:
    DEVICE = "cpu"

MODEL_TYPE = "gan"
USE_SUBSET = True  # True for Phase 1 (20%). False for Phase 2 (100%).

# --- EXPERIMENT HYPERPARAMETERS ---
EXP_NAME = "DCGAN_Exp4_Latent128_Lr1e-4"
EPOCHS = 50        
BATCH_SIZE = 128
LR = 0.0002
LATENT_DIM = 128

print(f"=== STARTING DCGAN EXPERIMENT ON DEVICE: {DEVICE} ===")

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
# Train the DCGAN
# =================================================================
generator_dcgan, total_time, avg_epoch_time = train_gan(
    train_loader,
    num_epochs=EPOCHS,
    device=DEVICE,
    lr=LR,
    latent_dim=LATENT_DIM
)

# =================================================================
# Save Model and create Visual Grid
# =================================================================
save_model(generator_dcgan, f"weights_{EXP_NAME}.pth")
generate_and_save_visual_grid(
    model=generator_dcgan, 
    device=DEVICE, 
    latent_dim=LATENT_DIM, 
    model_type=MODEL_TYPE, 
    filename=f"grid_{EXP_NAME}.png"
)

# =================================================================
# Evaluation
# =================================================================
# Specific wrapper for DCGAN, which requires noise with shape (batch_size, 100, 1, 1)
class DCGANGeneratorWrapper(torch.nn.Module):
    def __init__(self, gan_model, latent_dim=100):
        super().__init__()
        self.gan_model = gan_model
        self.latent_dim = latent_dim
        
    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(DEVICE)
        return self.gan_model(z)

dcgan_for_evaluation = DCGANGeneratorWrapper(generator_dcgan, latent_dim=LATENT_DIM)

print("\n=== STARTING DCGAN EVALUATION ===")
all_fid, all_kid, all_is, all_lpips = run_full_evaluation(model_type=MODEL_TYPE, generator=dcgan_for_evaluation, real_dataloader=test_loader, device=DEVICE)

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
    extra_notes="Standard DCGAN Phase 1",
    all_fid=all_fid, all_kid=all_kid, all_is=all_is, all_lpips=all_lpips,
    total_time=total_time, avg_epoch_time=avg_epoch_time
)

print("=== DCGAN EXPERIMENT FULLY COMPLETED! ===")
