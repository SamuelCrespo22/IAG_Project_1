import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# Assume que importas a classe do dataset fornecida pelos professores
# from provided_codebase import ArtBenchDataset 

# 1. Definir as transformações (Exemplo para GANs/Diffusion: [-1, 1])
transform_gan = transforms.Compose([
    transforms.ToTensor(), # Converte para [0, 1] e formato (C, H, W)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Converte para [-1, 1]
])

# 2. Carregar o dataset usando a classe fornecida
# Usa o subset de 20% para a fase inicial de testes
dataset_train = ArtBenchDataset(root='./data', train=True, transform=transform_gan, subset=True)

# 3. Criar o DataLoader (cria os blocos [N, C, H, W])
dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2)