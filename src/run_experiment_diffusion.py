import torch
from dataset import get_dataloader
from train_diffusion import train_diffusion
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

MODEL_TYPE = "diffusion"
USE_SUBSET = True  # True for Phase 1 (20%).

# --- EXPERIMENT HYPERPARAMETERS ---
EXP_NAME = "Diffusion_Exp10_T1000_E500_Lr1e-4"
EPOCHS = 500       
BATCH_SIZE = 64
LR = 1e-4
T_STEPS = 1000    
TIME_DIM = 128

print(f"=== STARTING DIFFUSION EXPERIMENT ON DEVICE: {DEVICE} ===")

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
# Train Diffusion Model
# =================================================================
# train_diffusion returns both the U-Net architecture and the DDPM process class

unet, ddpm, total_time, avg_epoch_time = train_diffusion(
    train_loader,
    num_epochs=EPOCHS,
    device=DEVICE,
    lr=LR,
    timesteps=T_STEPS 
)


# =================================================================
# Save Model and create Visual Grid
# =================================================================
# Save the U-Net weights, as it contains all the learnable parameters
save_model(unet, f"weights_{EXP_NAME}.pth")

# =================================================================
# Evaluation
# =================================================================
# We create a clever wrapper to bridge the DDPM class with our utils and evaluation scripts
class DiffusionGeneratorWrapper(torch.nn.Module):
    def __init__(self, ddpm_process):
        super().__init__()
        self.ddpm = ddpm_process
        
    def forward(self, x):
        # evaluate.py passes an int (batch_size)
        # utils.py passes a tensor (z)
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
        else:
            batch_size = x
            
        # Generate images using the reverse diffusion process
        return self.ddpm.sample(batch_size)

diffusion_for_evaluation = DiffusionGeneratorWrapper(ddpm)

generate_and_save_visual_grid(
    model=diffusion_for_evaluation, 
    device=DEVICE, 
    latent_dim=1, # Not strictly used by DDPM.sample, but required by the utils function signature
    model_type=MODEL_TYPE, 
    filename=f"grid_{EXP_NAME}.png"
)

print("\n=== STARTING DIFFUSION EVALUATION ===")
all_fid, all_kid, all_is, all_lpips = run_full_evaluation(model_type=MODEL_TYPE, generator=diffusion_for_evaluation, real_dataloader=test_loader, device=DEVICE)

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
    latent_dim=TIME_DIM,
    extra_notes=f"Diffusion with T={T_STEPS}",
    all_fid=all_fid, all_kid=all_kid, all_is=all_is, all_lpips=all_lpips,
    total_time=total_time, avg_epoch_time=avg_epoch_time
)

print("=== DIFFUSION EXPERIMENT FULLY COMPLETED! ===")
