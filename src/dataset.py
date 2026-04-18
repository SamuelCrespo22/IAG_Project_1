from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

from artbench_local_dataset import resolve_dataset_splits

# Create a PyTorch Dataset wrapper around the Hugging Face dataset.
class ArtBenchPyTorchWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Hugging Face returns a dictionary, we want a tuple (image, label).
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloader(model_type="gan", batch_size=64, use_subset=True, subset_fraction=0.2, data_source="hf", kaggle_root="../ArtBench-10", split="train"):
    """
    model_type: "vae", "gan", "wgan", or "diffusion".
    use_subset: If True, use only 20% of data (for Phase 1). Only applies to 'train' split.
    data_source: "hf" (does automatic download) or "kaggle" (reads from local data).
    kaggle_root: Only used if data_source="kaggle". Indicates where the folder is extracted.
    split: "train" or "test".
    """

    print(f"\n[DataLoader] Initializing for split='{split}', model_type='{model_type}'...")
    
    ds_dict = resolve_dataset_splits(
        dataset_id="philschmid/artbench-10",
        seed=42, 
        dataset_source=data_source, 
        kaggle_root=kaggle_root
    )
    
    if split not in ds_dict:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(ds_dict.keys())}")
    
    hf_dataset = ds_dict[split]

    # ====================================================
    # Define Transformations.
    # ====================================================

    if model_type == "vae": # VAE (uses sigmoid) Dataloader must be in [0, 1].
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        print(f"[DataLoader] Applying normalization to [0, 1] for {model_type.upper()}.")

    elif model_type in ["gan", "wgan", "diffusion"]: # GAN/WGAN (uses tanh) and Diffusion (noise needs [-1, 1]) Dataloader must be in [-1, 1].
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"[DataLoader] Applying normalization to [-1, 1] for {model_type.upper()}.")

    else:
        raise ValueError("invalid model_type!")

    # ====================================================
    # 20% Subset Logic (applies only to train split).
    # ====================================================
    
    if use_subset and split == "train":
        num_samples = int(len(hf_dataset) * subset_fraction)
        # Fix seed to guarantee the same samples for training.
        indices = np.random.RandomState(42).permutation(len(hf_dataset))[:num_samples]
        hf_dataset = hf_dataset.select(indices)
        print(f"[DataLoader] Using subset of {len(hf_dataset)} samples ({subset_fraction*100:.0f}% of train split).")
    else:
        print(f"[DataLoader] Using full '{split}' split with {len(hf_dataset)} samples.")

    # ====================================================
    # Create PyTorch Dataset and DataLoader.
    # ==================================================== 

    pytorch_dataset = ArtBenchPyTorchWrapper(hf_dataset, transform=transform)

    # Shuffle must be True for train and False for test to guarantee consistence in evaluation.
    dataloader = DataLoader(
        pytorch_dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        num_workers=0,
        pin_memory=True, # put data into a fixed space in RAM. Faster transfer of data.
        drop_last=(split == "train") # Drop last in train for stability.
    )
    
    return dataloader