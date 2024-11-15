# Conditional Diffusion Model for Optical Design Generation

This project implements a conditional diffusion model to generate optical designs based on specified power transmittance and wavelength conditions.

## Prerequisites

- Python 3.11.3
- NVIDIA GPU with CUDA support
- PyTorch with appropriate CUDA version

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv controldiff-venv
   source controldiff-venv/bin/activate  # On Windows use: controldiff-venv\Scripts\activate
   ```

3. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Replace `cu118` with your CUDA version if different.

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Setup Name Convention

Each experiment requires a unique setup name that will be used across all training and evaluation steps. This is controlled by the `--setting_name` parameter (e.g., "Setup1", "Setup2", etc.). Make sure to use the same setup name consistently across all steps of your experiment.

### Training the DL Estimator (Power Predictor)

Before training the diffusion model, we need to train a deep learning estimator that predicts power values given design images and wavelengths. This model serves as a fast alternative to the Maxwell FDFD simulator.

To train the estimator:

```bash
python src/train_classifier.py \
    --setting_name "Setup1" \
    --empower_lr 1e-4 \
    --empower_batch_size 64 \
    --clf_niters 100 \
    --val_every 5 \
    --patience 30 \
    --weight_decay 1e-4
```

Key parameters:
- `--setting_name`: Unique identifier for your experiment setup
- `--empower_lr`: Learning rate for the estimator model
- `--empower_batch_size`: Batch size for training
- `--clf_niters`: Number of epochs for training
- `--val_every`: Validation frequency (in epochs)
- `--patience`: Early stopping patience
- `--weight_decay`: Weight decay for regularization

The trained model will be saved in `output/{setting_name}/results/` with checkpoints and training history.

### Training the Diffusion Model

Training parameters are configured in `src/scripts/run_train.bat`. Modify this file to set your experiment parameters:

```batch
python src/main.py ^
--setting_name "Setup1" ^
--niters 200000 ^
--resume_niter 0 ^
--train_batch_size 16 ^
--train_lr 1e-4 ^
--sample_every 1000 ^
--save_every 10000
```

Key parameters to modify:
- `--setting_name`: Must match the setting name used in previous steps
- `--niters`: Total number of training iterations
- `--resume_niter`: Iteration to resume from (0 for new training)
- `--train_batch_size`: Batch size for training
- `--train_lr`: Learning rate
- `--sample_every`: Frequency of sample generation
- `--save_every`: Frequency of model checkpointing

Then run the training:
```bash
src/scripts/run_train.bat
```

### Generating Samples

To generate samples, you can use weights from any specific iteration by modifying the `run_train.bat` file:

1. Set both `--niters` and `--resume_niter` to the desired iteration number:
   ```batch
   --niters 150000 ^
   --resume_niter 150000
   ```

2. Run the script:
   ```bash
   src/scripts/run_train.bat
   ```

Generated samples will be saved in `output/{setting_name}/results/generated_samples/`.

### Evaluating Generated Samples

To evaluate the quality of generated samples:

```bash
python src/evaluate.py --setting_name "Setup1"
```

This will:
1. Load the trained DL Estimator
2. Process all generated samples
3. Calculate power predictions
4. Compare with ground truth values
5. Generate evaluation metrics and statistics

Results will be saved to `output/{setting_name}/results/evaluation/predicted_powers{setup_number}.csv`

## Project Structure

```
CONTROLDIFF/
├── controldiff-venv/
├── data/
│   ├── image/                        # Directory containing 32x32 design images (.tiff files)
│   ├── 0130_40_generated_wave_array2.csv
│   └── 0130_40_generated_power_array2.csv
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── aux_net.py
│   │   ├── empower.py
│   │   ├── ResNet_embed.py
│   │   └── unet.py
│   ├── scripts/
│   │   └── run_train.bat
│   ├── output/
│   │   ├── embed_models/
│   │   │   ├── ckpt_ResNet34_embed_epoch_200_seed_0.pth
│   │   │   └── ckpt_net_y2h_epoch_500_seed_0.pth
│   │   └── {setting_name}/
│   │       ├── results/
│   │       │   ├── generated_samples/
│   │       │   ├── evaluation/
│   │       │   ├── classifier/
│   │       │   └── model-{milestone}.pt
│   ├── diffusion.py
│   ├── evaluate.py
│   ├── main.py
│   ├── opts.py
│   ├── train_classifier.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Model Architecture

The model uses a U-Net architecture with attention mechanisms and incorporates two conditions:
- Power Transmittance
- Wavelength

The conditions are embedded and concatenated with time step embeddings to condition the diffusion process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns regarding this project, please contact:

Jaewon Kim - jwk0302@korea.ac.kr
