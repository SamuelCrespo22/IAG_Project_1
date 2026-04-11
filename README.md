# Projeto 1: ArtBench Generative Modeling

Este repositório contém o código e os resultados do Projeto 1 da disciplina de Inteligência Artificial Generativa (2025/2026), da Universidade de Coimbra. 

O objetivo principal deste projeto é conceber, treinar e avaliar diferentes famílias de modelos generativos na criação de imagens de obras de arte, utilizando o dataset **ArtBench-10** (resolução 32x32 RGB).

**Autores:**
* [Nome do Aluno 1] - [Número de Estudante 1]
* [Nome do Aluno 2] - [Número de Estudante 2]

---

## 🏗️ Modelos Implementados
Foram implementadas e comparadas três famílias de modelos:
1. **Autoencoders:** Variational Autoencoder (VAE)
2. **GANs:** Deep Convolutional GAN (DCGAN)
3. **Modelos de Difusão:** [Nome do modelo de difusão escolhido]

---

## 📊 Protocolo de Avaliação
Os modelos foram inicialmente iterados usando um subset de 20% dos dados. A avaliação final no dataset completo (50.000 imagens) segue um protocolo rigoroso:
* Geração de **5.000 amostras** por modelo.
* Comparação com **5.000 imagens reais**.
* Cálculo das métricas **FID** (Fréchet Inception Distance) e **KID** (Kernel Inception Distance).
* Repetição de todo o processo de avaliação usando **10 *seeds* aleatórias** diferentes para robustez estatística.

---

## ⚙️ Instalação e Configuração

1. Clona este repositório:
   ```bash
   git clone [https://github.com/teu-utilizador/artbench-generative-modeling.git](https://github.com/teu-utilizador/artbench-generative-modeling.git)
   cd artbench-generative-modeling