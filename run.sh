# Set environment variable to bypass proxy for specific domains
export no_proxy='huggingface.co'

# Install necessary libraries from specified directories
pip install /opt/huawei/dataset/all/code/pytorch-lightning torch==2.1.0
pip install -r /opt/huawei/dataset/all/code/clip-lightning/requirements-hr.txt

# Log in to Weights & Biases (wandb) with the provided token
wandb login 7b6d07bce88338250492397fd23e6cd84e8efdd2

# Copy datasets or models to the local cache directory
cp -r /opt/huawei/dataset/all/huggingface ~/.cache/
cp -r /opt/huawei/dataset/all/code ./

# Navigate to the working directory where the code is located
cd ./code/lightning-npu

# Update the repository to the latest version from the origin
git pull origin

# Execute the python script for training on CIFAR100 with specific configurations
python cifar100_224x224_fp16.py
