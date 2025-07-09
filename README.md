# ReVolVE: Neural Reconstruction of Volumes for Visualization Enhancement of Direct Volume Rendering

## Environment Setup
To set up the default Conda environment for this project, please execute the following commands:
```
# Create and activate the Conda environment
conda create -n revolve python=3.9
conda activate revolve

# Install CUDA Toolkit (version 11.6.2)
conda install -c nvidia/label/cuda-11.6.2 cuda-toolkit

# Install PyTorch with CUDA 11.6 support
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install additional dependencies
python -m pip install tqdm configargparse imageio kornia opencv-python scipy plyfile scikit-image numpy==1.24
```

##
