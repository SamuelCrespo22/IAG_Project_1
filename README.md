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

---

## How to Run

All the experiment configurations (e.g., hyper-parameters, data source, subset usage) can be directly modified within the respective `src/run_experiment_*.py` scripts. Evaluation configurations can be changed in `src/evaluate.py`.

To run a specific model experiment, execute one of the following commands from the root directory:

**1. VAE (Variational Autoencoder):**

```bash
python src/run_experiment_vae.py
```

**2. DCGAN (Deep Convolutional GAN):**

```bash
python src/run_experiment_dcgan.py
```

**3. WGAN (Wasserstein GAN):**

```bash
python src/run_experiment_wgan.py
```

**4. DDPM (Denoising Diffusion Probabilistic Model):**

```bash
python src/run_experiment_diffusion.py
```

**Note:** This will train the model and also do the evaluation. You should also have the Artbench-10 folder with the dataset inside the root folder for this to work. You can download it here: https://www.kaggle.com/datasets/alexanderliao/artbench10. Then, put the folder in the root folder and rename it to "ArtBench-10".
