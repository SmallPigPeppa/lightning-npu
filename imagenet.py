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

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        # Training data transformations
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Test data transformations (less augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # Load the dataset with appropriate transformations
        self.imagenet_train = torchvision.datasets.ImageFolder(root=DATASET_PATH + '/train', transform=self.train_transform)
        self.imagenet_val = torchvision.datasets.ImageFolder(root=DATASET_PATH + '/val', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # Use the validation dataset
        return DataLoader(self.imagenet_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size)


class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = resnet50(pretrained=True)
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
    data_module = ImageNetDataModule(batch_size=64)
    model = ResNet50Classifier()
    trainer = Trainer(accelerator='npu', devices='0,1', max_epochs=5, precision=16)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
