import torch
import torchvision
import torchvision.transforms as transforms
import lightning as pl
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning.loggers import WandbLogger
import timm

# DATASET_PATH = '/home/ma-user/work/dataset/all/torch_ds'
DATASET_PATH = '/opt/huawei/dataset/all/torch_ds'


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        # Training data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        # Test data transformations (less augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])

    def setup(self, stage=None):
        # 下载并应用适当的转换
        self.cifar100_train = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=True, download=True,
                                                            transform=self.train_transform)
        self.cifar100_test = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=False, download=True,
                                                           transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    def val_dataloader(self):
        # 使用测试集作为验证集
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, pin_memory=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, pin_memory=True, num_workers=8)


class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=100):
        super().__init__()
        # self.model = resnet50(pretrained=True)
        self.model = timm.create_model('resnet50', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.acc = Accuracy(num_classes=num_classes, task="multiclass", top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log('test/loss', loss)
        self.log('test/acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


def main():
    data_module = CIFAR100DataModule(batch_size=256)
    model = ResNet50Classifier()
    wandb_logger = WandbLogger(name='cifar100-r50', project='lightning-npu', entity='pigpeppa', offline=True)
    trainer = Trainer(accelerator='npu', logger=wandb_logger, max_epochs=50, precision=16, num_nodes=2)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
