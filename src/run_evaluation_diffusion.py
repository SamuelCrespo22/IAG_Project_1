import torch
from dataset import get_dataloader
from models.diffusion import UNet32, DDPM
from evaluate import run_full_evaluation
from utils import log_experiment_to_csv, generate_and_save_visual_grid

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

# --- EVALUATION HYPERPARAMETERS ---
EXP_NAME = "Diffusion_Evaluation_Run"
WEIGHTS_PATH = "weights_Diffusion_Exp14_T500_E500_LrNotFixed.pth" # Atualiza isto com o nome do teu ficheiro de pesos
BATCH_SIZE = 64
T_STEPS = 500  

print(f"=== STARTING DIFFUSION EVALUATION ON DEVICE: {DEVICE} ===")

# =================================================================
# Load Data (Apenas o Test Split é necessário para a avaliação)
# =================================================================
test_loader = get_dataloader(
    model_type=MODEL_TYPE,
    batch_size=BATCH_SIZE,
    use_subset=False,
    split="test",
    data_source="kaggle"
)

# =================================================================
# Initialize Model & Load Weights
# =================================================================
unet = UNet32().to(DEVICE)

try:
    unet.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print(f"[INFO] Successfully loaded weights from {WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
    print("Please specify the correct WEIGHTS_PATH.")
    exit(1)

ddpm = DDPM(unet, T=T_STEPS, device=DEVICE)

# =================================================================
# Evaluation Wrapper
# =================================================================
class DiffusionGeneratorWrapper(torch.nn.Module):
    def __init__(self, ddpm_process):
        super().__init__()
        self.ddpm = ddpm_process
        
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
        else:
            batch_size = x
        return self.ddpm.sample(batch_size)

diffusion_for_evaluation = DiffusionGeneratorWrapper(ddpm)

# (Opcional) Gerar uma grid visual rápida para confirmar que o modelo carregou bem
generate_and_save_visual_grid(
    model=diffusion_for_evaluation, 
    device=DEVICE, 
    latent_dim=1, 
    model_type=MODEL_TYPE, 
    filename=f"grid_eval_{EXP_NAME}.png"
)

# =================================================================
# Evaluation
# =================================================================
print("\n=== STARTING DIFFUSION EVALUATION ===")
all_fid, all_kid, all_is, all_lpips = run_full_evaluation(
    model_type=MODEL_TYPE, 
    generator=diffusion_for_evaluation, 
    real_dataloader=test_loader, 
    device=DEVICE
)

# =================================================================
# Log the results
# =================================================================
log_experiment_to_csv(
    csv_file="evaluation_results.csv",
    exp_name=EXP_NAME,
    model_type=MODEL_TYPE,
    epochs="N/A",            # Como é só avaliação, não temos o valor exato aqui
    batch_size=BATCH_SIZE,
    lr="N/A",                # Como é só avaliação, não temos o valor exato aqui
    latent_dim=128,
    extra_notes=f"Evaluation only. Weights: {WEIGHTS_PATH}",
    all_fid=all_fid, all_kid=all_kid, all_is=all_is, all_lpips=all_lpips,
    total_time_s=0, avg_epoch_time_s=0  # Não aplicável para avaliação
)

print("=== DIFFUSION EVALUATION FULLY COMPLETED! ===")
