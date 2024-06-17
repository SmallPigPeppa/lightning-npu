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

DATASET_PATH = '/home/ma-user/work/wenzhuoliu/torch_ds'


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        # 训练时使用的数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        # 测试时使用的数据增强（一般较少）
        self.test_transform = transforms.Compose([
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
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # 使用测试集作为验证集
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)


class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet50(pretrained=True)
        # self.model.conv1 = nn.Conv2d(3, self.model.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.acc = Accuracy(num_classes=num_classes, task="multiclass", top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


def main():
    data_module = CIFAR100DataModule(batch_size=64)
    model = ResNet50Classifier()
    trainer = Trainer(accelerator='npu', devices='0,1', max_epochs=5, strategy='deepspeed')
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
