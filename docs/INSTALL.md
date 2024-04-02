## Installation
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- Linux
- Python 3.9+
- PyTorch 2.2
- CUDA 12.1 or higher
- CMake 3.13.2 or higher
- [spconv](https://github.com/traveller59/spconv) 

### Basic Installation 

```bash
# basic python libraries
conda create --name centerpoint python=3.9
conda activate centerpoint
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/tianweiy/CenterPoint.git
cd CenterPoint
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_PATH=/usr/local/cuda-12.1
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Rotated NMS 
cd ROOT_DIR/det3d/ops/iou3d_nms
python setup.py build_ext --inplace

# Deformable Convolution (Optional and only works with old torch versions e.g. 1.1)
cd ROOT_DIR/det3d/ops/dcn
python setup.py build_ext --inplace
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 
