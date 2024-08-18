# from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger


# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import timm

model = timm.create_model('resnet50', pretrained=True)