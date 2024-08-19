# Disable proxy for specific domains
export no_proxy='huggingface.co'

# Install required Python packages from a given requirements file
pip install -r /opt/huawei/dataset/all/code/clip-lightning/requirements-hr.txt

# Log in to Weights & Biases for experiment tracking
wandb login 7b6d07bce88338250492397fd23e6cd84e8efdd2

wandb login

# Copy the Hugging Face datasets to the local cache directory
cp -r /opt/huawei/dataset/all/huggingface ~/.cache/

# Navigate to the project directory for training on the NPU
cd /opt/huawei/dataset/all/code/lightning-npu

# Update the local repository to ensure it's up-to-date
git pull origin

# Run a Python script for training a model on CIFAR-100 dataset with specific settings
python cifar100_224x224_fp16.py
