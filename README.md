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

### Training the Model

To train the model, use the provided training script:

```bash
run_train.bat
```

This script uses the parameters defined in `src/scripts/run_train.bat` to start the training process.

#### Important Parameters:

- `--niters`: Total number of iterations for training. Set this to your desired total training iterations.
- `--resume_niter`: Iteration number to resume training from. Used when continuing interrupted training.

To start a new training session:
```bash
--niters 200000 --resume_niter 0
```

To resume training from a specific iteration (e.g., 100000):
```bash
--niters 200000 --resume_niter 100000
```

### Generating Samples

To generate samples, you can use weights from any specific iteration:

1. Open the `src/scripts/run_train.bat` file.
2. Set both `--niters` and `--resume_niter` to the desired iteration number:
   ```bash
   --niters 150000 --resume_niter 150000
   ```
   This will use the weights from iteration 150000 for sampling.
3. Run the script:
   ```bash
   run_train.bat
   ```

The script will display a message indicating which iteration's weights are being used for sampling:
""Generating samples using weights: `model-{args.niters}`.pt"

Generated samples will be saved in the `output/Setup1/results/generated_samples/` directory.

Note: If you want to sample using weights from an iteration other than the final one, always set both `--niters` and `--resume_niter` to the same value of the desired iteration.

## Resuming Interrupted Sampling

If the sampling process is interrupted, you can resume from where it left off:

Simply run the `run_train.bat` script again.
The script will automatically detect the last sampled index and continue from there.

The sampling progress is saved in a JSON file located at `output/Setup1/results/sampling_progress.json`. This file keeps track of the last sampled index and the current image counter.

## Project Structure

The project structure is organized as follows:

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
│   │   ├── ResNet_embed.py
│   │   └── unet.py
│   ├── scripts/
│   │   ├── run_train.bat
│   ├── output/
│   │   ├── embed_models/
│   │   │   ├── ckpt_ResNet34_embed_epoch_200_seed_0.pth
│   │   │   └── ckpt_net_y2h_epoch_500_seed_0.pth
│   │   └── Setup1/
│   │       ├── results/
│   │       │   ├── generated_samples/
│   │       │   │   └── condition_{idx}_sample_{sample_idx}.tiff
│   │       │   ├── sample_{step}_grid.png
│   │       │   └── model-{milestone}.pt
│   │       └── log_loss_niters{train_num_steps}.txt
│   ├── diffusion.py
│   ├── ema_pytorch.py
│   ├── main.py
│   ├── opts.py
│   ├── train_net_for_label_embed.py
│   ├── trainer.py
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
