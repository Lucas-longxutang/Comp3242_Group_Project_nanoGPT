# Architectural Ablation and Ternary Weight Quantization in Small Transformers

**Team 22**
**Team Members:** Ruizhe Du (u7465990), Yunkai Xu (u7465735), Tom Bushell (u7473046), Madhav Krishnan (u7735537)

## Introduction

This project investigates the architectural dependencies and numerical precision limits of Transformer-based language models, using Andrej Karpathy’s `nanoGPT` as a foundational baseline. We established our baseline by training the model on the Tiny Shakespeare dataset. Through a systematic ablation study, we modified or removed key architectural components—such as positional encodings, normalization layers, residual connections, and attention heads—to observe their impact on training dynamics and text generation quality.

Furthermore, as an extension of model design and numerical precision, we implemented ternary weight quantization. Using a Straight-Through Estimator (STE) during the backward pass, we restricted the model weights strictly to the set {-1, 0, +1}. Our primary goal is to understand which components of the Transformer architecture are strictly essential for sequence modeling and to explore the performance trade-offs of extreme low-bit quantization in low-resource environments.

## Core Ablation Axes

To ensure a rigorous comparison, all experiments were conducted under strictly controlled conditions: the same dataset, optimizer, base model size, and training environment. We explored the design space across seven specific axes:

* **A. Positional Encoding:** Tested No Positional Encoding (NoPE), Rotary Positional Embedding (RoPE), and ALiBi.
* **B. Normalization:** Compared RMSNorm, Pre-LayerNorm (baseline), and Post-LayerNorm.
* **C. Attention Heads:** Evaluated the capacity distribution across 1, 4 (baseline), 8, and 12 attention heads.
* **D. Activation Functions:** Compared GELU (baseline) against ReLU and SwiGLU.
* **E. Residual Connections:** Tested the standard residual path (baseline), a scaled residual (1 / √L), and the complete removal of skip connections.
* **F. Context Length:** Examined the effects of reducing the context window to 128 tokens or expanding it to 512 tokens.
* **G. Quantization:** Implemented a BitNet-style ternary weight quantization layer ({-1, 0, 1}) with group-wise scaling.

## Hardware & Environment

All 64 experimental training runs were conducted and validated locally on the following hardware setup:
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
* **VRAM:** 8.0 GB Dedicated GPU Memory
* **Driver Version:** 571.96 (Windows Device Manager: 32.0.15.7196)
* **Average Training Time:** ~22 minutes per baseline run

## Repository Structure

* `01_prepare.ipynb`: Downloads and tokenizes the Tiny Shakespeare dataset, setting up the PyTorch DataLoader.
* `02_train.ipynb`: Contains the core PyTorch training loop and the integrated ablation framework where hyperparameters can be modified.
* `03_evaluvation.ipynb`: Evaluates final model metrics (Validation Loss & Perplexity) and qualitatively assesses generated text.
* `aggregator.py`, `grapher.py`, `runretrival.py`: Auxiliary Python scripts used for extracting multiple random-seed experiment data and generating professional analytical charts.
* `requirements.txt`: Python environment dependencies.
* `training_records/`: Organized directory containing aggregated CSV files of the ablation study results and Weights & Biases export data (e.g., `metrics.csv`, `ablation_summary.csv`) used for analysis and plotting.

## Installation

Ensure you have Python 3.8+ installed on your system. To leverage GPU acceleration with the local GPU, please visit the official websites below to configure your local CUDA environment and install the appropriate version of PyTorch compiled with **CUDA 12.1 or newer**:

* **PyTorch Official Website:** [https://pytorch.org/](https://pytorch.org/)
* **NVIDIA CUDA Toolkit Archive:** [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

Once your CUDA environment and GPU-enabled PyTorch are ready, install the remaining project dependencies:

pip install -r requirements.txt

## How to Run

To reproduce the baseline results or run specific architectural ablation/quantization experiments, follow the sequential steps below:

### 1. Data Preparation
Before starting any training, you must prepare and tokenize the dataset. Open and run all cells in the data preparation notebook:

01_prepare.ipynb

This script will download the Tiny Shakespeare dataset, process it at the character level, and save the tokenized shards ready for training.

### 2. Configure and Run Ablation Tests
The experimental configurations and parameters are managed directly within the training script.
1. Open `02_train.ipynb`.
2. Locate the parameter configuration section/variables at the top of the notebook or within the training setup cells.
3. Modify the specific architectural parameters or toggles to select your desired experiment axis (e.g., enabling ternary quantization `G1`, disabling positional encodings `A1`, or removing residual connections `E2`).
4. Run the notebook cells to execute the training pipeline.

### 3. Evaluation and Qualitative Analysis
Once a training run completes and the model weights are saved, you can analyze the performance metrics and test text generation by running:

03_evaluvation.ipynb

This notebook will calculate the final Validation Loss, Perplexity, and generate sample Shakespearean text for qualitative analysis.

## Acknowledgments

The baseline architecture for this project is built upon nanoGPT by Andrej Karpathy. The ternary quantization implementation draws inspiration from the methodology presented in *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* by Ma et al., 2024.