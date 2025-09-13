#!/bin/bash

ENV_NAME=ikinci_gorev_env

PYTHON_BIN=~/.local/python-3.10.12/bin/python3

$PYTHON_BIN -m venv ~/ARTEK/SYZ_25/$ENV_NAME

source ~/ARTEK/SYZ_25/$ENV_NAME/bin/activate

pip install --upgrade pip

pip install numpy==1.26.4

pip install scipy==1.15.3 scikit-learn matplotlib pandas

pip install pillow opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86

pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install tqdm seaborn

python -c "import torch; print('✅ CUDA OK' if torch.cuda.is_available() else '🚫 CUDA NOT AVAILABLE')"

