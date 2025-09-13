ENV_NAME="siniflandirma_env"
PYTHON_VERSION="3.10"

python${PYTHON_VERSION} -m venv $ENV_NAME
source $ENV_NAME/bin/activate

python -m pip install --upgrade "pip<25"

pip install --upgrade "tensorflow[and-cuda]==2.15.1"

pip install numpy==1.26.4 ml-dtypes==0.3.2
pip install opencv-python==4.11.0.45 pillow matplotlib scikit-learn pandas tqdm

echo "✅ $ENV_NAME kuruldu ve aktif. GPU kontrolü:"
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

