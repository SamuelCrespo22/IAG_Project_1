import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
import lpips

# === EVALUATION CONFIGURATIONS ===
NUM_SAMPLES = 5000
NUM_SEEDS = 10
KID_SUBSETS = 50
KID_SUBSET_SIZE = 100
BATCH_SIZE = 64 # Adjust based on GPU VRAM.

def prepare_images_for_metrics(images_tensor, model_type="gan"):
    """
    Converts tensors to uint8 [0, 255] format required by TorchMetrics.
    """
    if model_type in ["gan", "wgan", "diffusion"]: # [-1, 1] => [0, 1].
        images_0_to_1 = (images_tensor + 1.0) / 2.0
    elif model_type == "vae": # VAE is already in [0, 1].
        images_0_to_1 = images_tensor
    else:
        images_0_to_1 = images_tensor
        
    images_0_to_1 = torch.clamp(images_0_to_1, 0.0, 1.0)
    images_uint8 = (images_0_to_1 * 255).byte()
    return images_uint8

def run_full_evaluation(model_type, generator, real_dataloader, device="cuda"):
    print(f"[INFO] Starting evaluation with {NUM_SEEDS} seeds. This might take a while...")
    
    # Initialize LPIPS metric (uses pre-trained VGG network).
    # LPIPS is calculated using float tensors in [-1, 1], so we don't use the uint8 conversion
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval() # Evaluation mode (we don't want to train VGG)

    all_fid, all_kid, all_is, all_lpips = [], [], [], []

    for seed in range(NUM_SEEDS):
        print(f"\n--- Processing Seed {seed+1}/{NUM_SEEDS} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize TorchMetrics
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid = KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, subsets=KID_SUBSETS, normalize=False).to(device)
        inception = InceptionScore(normalize=False).to(device)
        
        # 1. PROCESS REAL IMAGES
        print("Extracting features from real images...")
        real_count = 0
        for real_batch, _ in tqdm(real_dataloader, leave=False):
            current_batch_size = real_batch.size(0)
            if real_count + current_batch_size > NUM_SAMPLES:
                real_batch = real_batch[:NUM_SAMPLES - real_count]
            
            real_batch = real_batch.to(device)
            real_images_uint8 = prepare_images_for_metrics(real_batch, model_type)
            
            fid.update(real_images_uint8, real=True)
            kid.update(real_images_uint8, real=True)
            
            real_count += real_images_uint8.size(0)
            if real_count >= NUM_SAMPLES:
                break
                
        # 2. GENERATE FAKE IMAGES
        print("Generating new images and extracting features...")
        fake_count = 0
        generated_batches_for_lpips = [] # Store a few batches for LPIPS
        
        generator.eval() # Ensure the model is not in training mode
        with torch.no_grad():
            while fake_count < NUM_SAMPLES:
                current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - fake_count)
                
                # Generate fake images using the wrapper generator
                fake_batch = generator(current_batch_size)
                
                # Store the first two batches in float format for LPIPS calculation
                if len(generated_batches_for_lpips) < 2:
                    generated_batches_for_lpips.append(fake_batch.clone())

                fake_images_uint8 = prepare_images_for_metrics(fake_batch, model_type)
                
                fid.update(fake_images_uint8, real=False)
                kid.update(fake_images_uint8, real=False)
                inception.update(fake_images_uint8)
                
                fake_count += current_batch_size

        # 3. CALCULATE LPIPS (DIVERSITY)
        # We compare images from batch 0 with batch 1
        if len(generated_batches_for_lpips) == 2:
            img1 = generated_batches_for_lpips[0]
            img2 = generated_batches_for_lpips[1]
            
            # LPIPS expects images in [-1, 1] range
            if model_type == "vae":
                img1 = img1 * 2.0 - 1.0
                img2 = img2 * 2.0 - 1.0
                
            # Ensure they have the same size
            min_len = min(img1.size(0), img2.size(0))
            lpips_score = loss_fn_vgg(img1[:min_len], img2[:min_len]).mean().item()
        else:
            lpips_score = 0.0

        # 4. COMPUTE SEED RESULTS
        print("Calculating final scores for this seed...")
        seed_fid = fid.compute().item()
        seed_kid_mean, seed_kid_std = kid.compute()
        seed_is_mean, seed_is_std = inception.compute()
        
        all_fid.append(seed_fid)
        all_kid.append(seed_kid_mean.item())
        all_is.append(seed_is_mean.item())
        all_lpips.append(lpips_score)
        
        print(f"Results Seed {seed}: FID={seed_fid:.2f} | KID={seed_kid_mean.item():.4f} | IS={seed_is_mean.item():.2f} | LPIPS={lpips_score:.4f}")
        
        # Clear GPU memory and reset metrics
        fid.reset(); kid.reset(); inception.reset()
        torch.cuda.empty_cache()

    # 5. AGGREGATE EVERYTHING (FOR THE REPORT TABLE)
    print("\n" + "="*40)
    print("FINAL RESULTS (MEAN ± STD DEV)")
    print("="*40)
    print(f"FID:   {np.mean(all_fid):.2f} ± {np.std(all_fid):.2f}  (Lower is better)")
    print(f"KID:   {np.mean(all_kid):.4f} ± {np.std(all_kid):.4f}  (Lower is better)")
    print(f"IS:    {np.mean(all_is):.2f} ± {np.std(all_is):.2f}  (Higher is better)")
    print(f"LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}  (Higher means more diversity)")
    print("="*40)

    return all_fid, all_kid, all_is, all_lpips