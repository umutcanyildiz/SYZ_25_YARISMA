#!/bin/bash

ENV_NAME="ikinci_gorev_env"
PYTHON_VERSION="3.10"

# Sistemdeki python3.10 yolunu otomatik bul
PYTHON_BIN=$(command -v python${PYTHON_VERSION})

if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Python ${PYTHON_VERSION} bulunamadı!"
  exit 1
fi

# Sanal ortamı scriptin çalıştırıldığı klasöre kur
VENV_DIR="$PWD/$ENV_NAME"

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade "pip<25"
pip install --upgrade numpy==1.26.4
pip install pylibjpeg==2.0.0 pylibjpeg-libjpeg==2.2.0
pip install scipy==1.15.3 scikit-learn matplotlib pandas rich timm pydicom
pip install pillow opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm seaborn

echo "✅ $ENV_NAME kuruldu ve aktif. CUDA kontrolü:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

