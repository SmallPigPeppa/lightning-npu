export no_proxy='huggingface.co'
pip install -r /opt/huawei/dataset/all/code/clip-lightning/requirements-hr.txt
wandb login 7b6d07bce88338250492397fd23e6cd84e8efdd2
cp -r /opt/huawei/dataset/all/huggingface ~/.cache/
cd /opt/huawei/dataset/all/code/lightning-npu
git pull origin
python cifar100_224x224_fp16.py
