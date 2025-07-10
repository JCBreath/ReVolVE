# ReVolVE: Neural Reconstruction of Volumes for Visualization Enhancement of Direct Volume Rendering

## Environment Setup
To set up the default Conda environment for this project, please execute the following commands:
```bash
# Create and activate the Conda environment
conda create -n revolve python=3.9
conda activate revolve

# Install CUDA Toolkit (version 11.6.2)
conda install -c nvidia/label/cuda-11.6.2 cuda-toolkit

# Install PyTorch with CUDA 11.6 support
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install additional dependencies
python -m pip install tqdm configargparse imageio kornia opencv-python scipy plyfile scikit-image lpips
python -m pip install numpy==1.24
```

## Dataset Preparation
This project follows the [NeRF](https://www.matthewtancik.com/nerf) synthetic dataset format based on Blender scenes.

A sample dataset *head* is included in this repository for demonstration purposes.

## Quick Start

### Training

To train a ReVolVE model, run:

```bash
python main.py --mode train --config configs/head.txt
```

The trained model will be saved by default at `log/head/head.th`

### Evaluation

After the model is trained, run:

```bash
python main.py --mode render_test --config configs/head.txt --ckpt log/head/head.th
```

The output will be saved at `log/head/render_output`

### Render with Enhancement

To render with enhancement enabled, run:

```bash
python main.py --mode render_test --enhanced --config configs/head.txt --ckpt log/head/head.th
```

Enhanced output will be saved at `log/head/enhanced_output`

### Export

**Export volume:**

```bash
python main.py --mode export_volume --config configs/head.txt --ckpt log/head/head.th
```

**Export mesh:**

```bash
python main.py --mode export_mesh --config configs/head.txt --ckpt log/head/head.th
```

Exported files (volume and mesh) will be saved by default at `log/head/exported_files`
