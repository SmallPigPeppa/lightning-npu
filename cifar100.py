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
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

DATASET_PATH = '/home/ma-user/work/wenzhuoliu/torch_ds'
max_epochs = 80
lr = 0.1
wd = 5e-4
batch_size = 256


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        # Training data transformations
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        # Test data transformations (less augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])

    def setup(self, stage=None):
        self.cifar100_train = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=True, download=True,
                                                            transform=self.train_transform)
        self.cifar100_test = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=False, download=True,
                                                           transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, pin_memory=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, pin_memory=True, num_workers=8)


class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
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
        params_to_optimize = self.model.parameters()

        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=lr,
            weight_decay=wd,
            momentum=0.9)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=max_epochs,
            warmup_start_lr=0.01 * lr,
            eta_min=0.01 * lr,
        )
        return [optimizer], [scheduler]


def main():
    data_module = CIFAR100DataModule(batch_size=batch_size)
    model = ResNet50Classifier()
    trainer = Trainer(accelerator='npu', devices='0,1', max_epochs=max_epochs, precision=16)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
