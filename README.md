# Project 1: ArtBench Generative Modeling

This repository contains the code and results of the first project from the Generative Artificial Intelligence course (2025/2026), University of Coimbra.

The main objective is to conceive, train and evaluate different families of generative models in the creation of artwork images, using the **ArtBench-10** dataset (32x32 RGB resolution).

**Authors:**

- Cláudio Catarino
- Samuel Crespo

---

## Implemented Models

1. **Autoencoders:** Variational Autoencoder (VAE);
2. **GANs:** Deep Convolutional GAN (DCGAN), Wasserstein GAN (WGAN);
3. **Diffusion Models:** Denoising Diffusion Probabilistic Model (DDPM).

---

## Evaluation Protocol

The models were initially tested with a 20% subset of the dataset. The best model was chosen by the lowest FID calculated. The chosen model was then trained on the full dataset and went through rigorous quantitative evaluation (FID, KID, IS, LPIPS) for 10 random seeds.
