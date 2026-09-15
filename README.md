# HealthcareFLProject - Privacy-Preserving Clinical NLP with Federated Learning

![License](https://img.shields.io/github/license/your-username/HealthcareFLProject) ![Language](https://img.shields.io/github/languages/top/your-username/HealthcareFLProject) ![Stars](https://img.shields.io/github/stars/your-username/HealthcareFLProject) ![Forks](https://img.shields.io/github/forks/your-username/HealthcareFLProject)

This project addresses the challenge of training NLP models on sensitive clinical data without centralization. It implements a Federated Learning pipeline using DistilBERT, optimized with LoRA for efficiency and secured with Differential Privacy.

---

## Federated Learning in Healthcare

Mapping where privacy, compression, and data imbalance collide in clinical NLP.

### Core Components

*   **Base Model:** `transformers` (DistilBERT, 66M params) for clinical text feature extraction.
*   **Adaptation:** `peft` (LoRA) to freeze base weights and train low-rank matrices, reducing trainable parameters by >99%.
*   **Privacy:** `opacus` (DP-SGD) for per-sample gradient clipping and noise addition.
*   **Orchestration:** `flwr` (Flower) to simulate hospital nodes, manage local training, and aggregate weights.
*   **Data Skew:** Dirichlet distribution (`α`) for non-IID data partitioning across simulated clients.
*   **Engine:** `PyTorch` for tensor operations and backpropagation.

### Usage

Run the main engine to orchestrate the federated learning pipeline.

```bash
# Run with default configuration
python main.py

# Specify a configuration file
python main.py --config configs/config.yaml

# Override specific parameters for a single run
python main.py --rank 4 --epsilon 8.0 --alpha 0.1

# Run without Differential Privacy for baseline comparison
python main.py --rounds 2 --epsilon null
```

## Local Setup

### Prerequisites

*   Python 3.8+

### Install Dependencies

This project uses OS-specific installation for PyTorch to ensure compatibility with your hardware (CUDA, MPS, or CPU).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fl-healthcare-repo.git
    cd fl-healthcare-repo
    ```

2.  **Install base dependencies:**
    ```bash
    pip install -r requirements/base.txt
    ```

3.  **Install OS-specific dependencies:**

    *   **Linux:**
        ```bash
        bash setup/setup_linux.sh
        ```
        (Installs packages listed in `requirements/linux.txt`)

    *   **macOS (Apple Silicon):**
        ```bash
        bash setup/setup_mac.sh
        ```
        (Installs packages listed in `requirements/mac.txt`)

    *   **Windows:**
        ```bash
        powershell -ExecutionPolicy Bypass -File setup/setup_windows.ps1
        ```
        (Installs packages listed in `requirements/windows.txt`)

### Run Simulations

*   **Single Run Simulation:**
    ```bash
    python main.py
    ```

*   **Benchmark Sweep:**
    ```bash
    python sweep.py
    ```
    (Uses settings from `configs/config.yaml`)

## Features

*   **Non-IID Data Partitioning:** Simulates realistic healthcare data distribution using Dirichlet partitioning with safeguards.
*   **Parameter-Efficient Fine-Tuning:** Employs LoRA (Low-Rank Adaptation) for DistilBERT, significantly reducing trainable parameters.
*   **Differential Privacy:** Integrates DP-SGD with per-sample gradient clipping and calibrated Gaussian noise for enhanced privacy.
*   **Multi-Client Federation:** Simulates a Flower federation with multiple hospital nodes for distributed learning.
*   **Comprehensive Metric Tracking:** Monitors key performance indicators including Accuracy, Precision, Recall, F1, MCC, and AUC-ROC.

## Tech Stack

*   **Core ML/DL:**
    *   [PyTorch](https://pytorch.org/): For tensor operations and automatic differentiation.
    *   [Transformers](https://huggingface.co/docs/transformers/index): Utilizes DistilBERT for feature extraction.
    *   [PEFT (LoRA)](https://github.com/huggingface/peft): For parameter-efficient fine-tuning via Low-Rank Adaptation.
*   **Federated Learning:**
    *   [Flower](https://flower.dev/): Orchestrates federated training across simulated clients.
*   **Differential Privacy:**
    *   [Opacus](https://opacus.ai/): Implements DP-SGD for privacy-preserving model updates.
*   **Data Handling:**
    *   [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution): Used to simulate non-IID data skew across clients.

## Usage

### Single Simulation Run

Execute a single federated learning simulation using `main.py`.

```bash

## Architecture

The HealthcareFLProject employs a layered architecture designed for privacy-preserving federated learning with parameter-efficient model adaptation.

### System Tracks

The project is organized into two core engineering tracks:

*   **Track A (Privacy & Data):**
    *   Handles non-IID data partitioning using a Dirichlet distribution.
    *   Implements safeguards for sample-count recovery.
    *   Manages per-sample gradient clipping and calibrated Gaussian DP-SGD noise.
    *   See `src/data/partition.py` and `src/privacy/dp_engine.py`.

*   **Track B (Model & Federation):**
    *   Focuses on parameter-efficient fine-tuning of DistilBERT using low-rank adapters (LoRA).
    *   Orchestrates federated training across multiple clients using Flower.
    *   Tracks comprehensive metrics including Accuracy, Precision, Recall, F1, MCC, and AUC-ROC.
    *   See `src/models/lora_model.py` and `src/federated/`.

### Architectural Layers

| Layer         | Tool/Component                               | Role                                                    |
| :------------ | :------------------------------------------- | :------------------------------------------------------ |
| Engine        | `PyTorch`                                    | Core tensor operations and backpropagation.             |
| Base Model    | `transformers` (DistilBERT, 66M params)      | Feature extraction from clinical text.                  |
| Adaptation    | `peft` (LoRA)                                | Trains low-rank matrices (`ΔW = BA`), reducing trainable parameters significantly. |
| Privacy       | `opacus` (DP-SGD)                            | Applies per-sample gradient clipping and Gaussian noise. |
| Orchestration | `flwr` (Flower)                              | Simulates multi-client federations and manages aggregation. |
| Data Skew     | Dirichlet distribution (`α`)                 | Partitions data non-IID across simulated hospitals.     |

## Repository Structure

This repository is organized to facilitate clear development and experimentation with federated learning in healthcare.

*   **`configs/`**: Stores experiment configuration files.
    *   `config.yaml`: Main configuration for sweep parameters (e.g., rounds, epsilon, alpha).
*   **`requirements/`**: Contains dependency lists for different operating systems.
    *   `base.txt`: OS-agnostic Python packages.
    *   `linux.txt`: Linux-specific dependencies.
    *   `mac.txt`: macOS-specific dependencies.
    *   `windows.txt`: Windows-specific dependencies.
*   **`setup/`**: Scripts for setting up the development environment.
    *   `setup_linux.sh`: Linux setup script.
    *   `setup_windows.ps1`: Windows setup script.
    *   `setup_mac.sh`: macOS setup script.
*   **`src/`**: Core source code for the project.
    *   **`data/`**: Data handling and partitioning logic.
        *   `partition.py`: Implements Dirichlet-based non-IID data partitioning.
    *   **`models/`**: Model definitions and implementations.
        *   `lora_model.py`: Defines a DistilBERT model with LoRA integration.
    *   **`privacy/`**: Modules for privacy-preserving techniques.
        *   `dp_engine.py`: Wrapper for Opacus for differential privacy (clipping and noise).
    *   **`federated/`**: Federated learning components.
        *   `client.py`: Flower client implementation (local DP-SGD training, metrics).
        *   `server.py`: Flower server implementation (FedAvg aggregation).
*   **`tests/`**: Unit tests for various project components.
    *   Includes tests for data partitioning, models, and DP engines.
*   **`README.md`**: The main project README file.
*   **`main.py`**: Entry point for running a single federated learning pipeline simulation.
*   **`sweep.py`**: Script for running benchmark sweeps across different parameters (r, ε, α).

## 7. Status

*   **Core Pipelines Functional:** Data partitioning, DP-SGD noise engine, LoRA fine-tuning, and Flower federation simulation are implemented and operational.
*   **Unit Tests Passing:** All zero-dependency unit tests for key components (`partition.py`, `dp_engine.py`, `lora_model.py`, `server.py`) are verified.
*   **End-to-End Simulation Validated:** The complete federated learning workflow has been successfully simulated.
*   **Grid Sweep Infrastructure:** The system supports automated grid sweeps for hyperparameter tuning.
*   **Analysis Visualization:** A configured notebook (`notebooks/analysis.ipynb`) is ready for visualizing benchmark results.

---

*This README was generated by [DevDoq](https://devdoq.com)*