from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torch
import numpy as np

def run_evaluation(generator, real_dataloader, device, num_runs=10):
    fid_scores = []
    kid_means = []
    
    for run in range(num_runs):
        # 1. Definir seed diferente para cada run 
        torch.manual_seed(run)
        
        # 2. Inicializar métricas 
        # (subset_size=100 para o KID conforme exigido) 
        fid = FrechetInceptionDistance(feature=64).to(device)
        kid = KernelInceptionDistance(subset_size=100).to(device)
        
        # 3. Processar 5000 imagens reais
        real_count = 0
        for real_batch in real_dataloader:
            if real_count >= 5000:
                break
            # torchmetrics espera imagens uint8 [0, 255]
            real_batch = ((real_batch + 1) / 2 * 255).to(torch.uint8).to(device) 
            fid.update(real_batch, real=True)
            kid.update(real_batch, real=True)
            real_count += real_batch.size(0)

        # 4. Gerar 5000 imagens falsas
        fake_count = 0
        while fake_count < 5000:
            batch_size = min(100, 5000 - fake_count)
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake_batch = generator(noise)
            # Converter de volta para [0, 255] uint8
            fake_batch = ((fake_batch + 1) / 2 * 255).to(torch.uint8).to(device)
            
            fid.update(fake_batch, real=False)
            kid.update(fake_batch, real=False)
            fake_count += batch_size

        # 5. Calcular resultados desta repetição
        fid_score = fid.compute().item()
        kid_mean, kid_std = kid.compute() # KID devolve média e std dos 50 subsets
        
        fid_scores.append(fid_score)
        kid_means.append(kid_mean.item())
        
        # Reset para a próxima seed
        fid.reset()
        kid.reset()

    # 6. Reportar estatísticas finais sobre as 10 repetições
    print(f"FID Final: Média = {np.mean(fid_scores):.4f}, Desvio Padrão = {np.std(fid_scores):.4f}")
    print(f"KID Final: Média = {np.mean(kid_means):.4f}, Desvio Padrão = {np.std(kid_means):.4f}")

# Exemplo de chamada:
# run_evaluation(generator, dataloader, device)