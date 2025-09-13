#!/bin/bash

ENV_NAME="siniflandirma_env"
PYTHON_VERSION="3.10"

# Sistemdeki python3.10 yolunu otomatik bul
PYTHON_BIN=$(command -v python${PYTHON_VERSION})

if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Python ${PYTHON_VERSION} bulunamadı!"
  exit 1
fi

# Sanal ortamı çalıştırıldığın klasöre kur
VENV_DIR="$PWD/$ENV_NAME"

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade "pip<25"
pip install --upgrade "tensorflow[and-cuda]==2.15.1"
pip install numpy==1.26.4 ml-dtypes==0.3.2
pip install opencv-python==4.11.0.45 pillow matplotlib scikit-learn pandas tqdm

echo "✅ $ENV_NAME kuruldu ve aktif. GPU kontrolü:"
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

