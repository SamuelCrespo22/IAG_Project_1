import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
import lpips
import ssl
import warnings

warnings.filterwarnings("ignore")

# Workaround for macOS.
ssl._create_default_https_context = ssl._create_unverified_context

# === EVALUATION CONFIGURATIONS ===
NUM_SAMPLES = 5000      # 1000 for Phase 1. Change to 5000 for Phase 2.
NUM_SEEDS = 10          # 3 seeds for Phase 1. Change to 10 for Phase 2.
KID_SUBSETS = 50
KID_SUBSET_SIZE = 100
BATCH_SIZE = 64

def prepare_images_for_metrics(images_tensor, model_type="gan"):
    """
    Converts tensors to uint8 [0, 255] format required by TorchMetrics.
    """
    if model_type in ["gan", "wgan", "diffusion"]:   # [-1, 1] => [0, 1].
        images_0_to_1 = (images_tensor + 1.0) / 2.0
    elif model_type == "vae":                        # VAE already in [0, 1].
        images_0_to_1 = images_tensor
    else:
        images_0_to_1 = images_tensor
        
    images_0_to_1 = torch.clamp(images_0_to_1, 0.0, 1.0)
    images_uint8 = (images_0_to_1 * 255).byte()
    return images_uint8

def run_full_evaluation(model_type, generator, real_dataloader, device="cuda"):
    print(f"[INFO] Starting evaluation with {NUM_SEEDS} seeds.")
    
    # Initialize LPIPS metric (uses pre-trained VGG network).
    # LPIPS uses float tensors in [-1, 1], so don't use the uint8 conversion.
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()

    # TorchMetrics requires float64.
    # use reset_real_features=False to save the computer from processing real images repeatedly.
    metric_device = "cpu" if str(device) == "mps" else device
    
    fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(metric_device)
    kid = KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, subsets=KID_SUBSETS, normalize=False, reset_real_features=False).to(metric_device)
    inception = InceptionScore(normalize=False).to(metric_device)

    all_fid, all_kid, all_is, all_lpips = [], [], [], []

    for seed in range(NUM_SEEDS):
        print(f"\nProcessing Seed {seed+1}/{NUM_SEEDS}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # ======================================================
        # Process real images
        # ======================================================
        if seed == 0:
            print("Extracting features from real images.")
            real_count = 0
            for real_batch, _ in tqdm(real_dataloader, leave=False):
                current_batch_size = real_batch.size(0)
                if real_count + current_batch_size > NUM_SAMPLES:
                    real_batch = real_batch[:NUM_SAMPLES - real_count]
                
                real_batch = real_batch.to(device)
                real_images_uint8 = prepare_images_for_metrics(real_batch, model_type)
                
                fid.update(real_images_uint8.to(metric_device), real=True)
                kid.update(real_images_uint8.to(metric_device), real=True)
                
                real_count += real_images_uint8.size(0)
                if real_count >= NUM_SAMPLES:
                    break
                
        # ======================================================
        # Generate fake images
        # ======================================================
        print("Generating new images and extracting features.")
        fake_count = 0
        generated_batches_for_lpips = []
        
        generator.eval()
        with torch.no_grad():
            while fake_count < NUM_SAMPLES:
                current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - fake_count)
                
                fake_batch = generator(current_batch_size)
                
                if len(generated_batches_for_lpips) < 2:
                    generated_batches_for_lpips.append(fake_batch.clone())

                fake_images_uint8 = prepare_images_for_metrics(fake_batch, model_type)
                
                fid.update(fake_images_uint8.to(metric_device), real=False)
                kid.update(fake_images_uint8.to(metric_device), real=False)
                inception.update(fake_images_uint8.to(metric_device))
                
                fake_count += current_batch_size

        if len(generated_batches_for_lpips) == 2:
            img1 = generated_batches_for_lpips[0]
            img2 = generated_batches_for_lpips[1]
            
            # LPIPS expects images in [-1, 1] range.
            if model_type == "vae":
                img1 = img1 * 2.0 - 1.0
                img2 = img2 * 2.0 - 1.0
                
            min_len = min(img1.size(0), img2.size(0))
            lpips_score = loss_fn_vgg(img1[:min_len], img2[:min_len]).mean().item()
        else:
            lpips_score = 0.0

        print("Calculating final scores for this seed.")
        seed_fid = fid.compute().item()
        seed_kid_mean, seed_kid_std = kid.compute()
        seed_is_mean, seed_is_std = inception.compute()
        
        all_fid.append(seed_fid)
        all_kid.append(seed_kid_mean.item())
        all_is.append(seed_is_mean.item())
        all_lpips.append(lpips_score)
        
        print(f"Results Seed {seed + 1}: FID={seed_fid:.2f} | KID={seed_kid_mean.item():.4f} | IS={seed_is_mean.item():.2f} | LPIPS={lpips_score:.4f}")
        
        fid.reset(); kid.reset(); inception.reset()
        if str(device) == "cuda":
            torch.cuda.empty_cache()
        elif str(device) == "mps":
            torch.mps.empty_cache()

    print("\n" + "="*40)
    print("Final Results (MEAN ± STD)")
    print("="*40)
    print(f"FID:   {np.mean(all_fid):.2f} ± {np.std(all_fid):.2f}")
    print(f"KID:   {np.mean(all_kid):.4f} ± {np.std(all_kid):.4f}")
    print(f"IS:    {np.mean(all_is):.2f} ± {np.std(all_is):.2f}")
    print(f"LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}  (Higher => more diversity)")
    print("="*40)

    return all_fid, all_kid, all_is, all_lpips