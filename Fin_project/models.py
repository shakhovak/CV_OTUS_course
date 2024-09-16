import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, output2, label):
        cosine_sim = F.cosine_similarity(output1, output2)
        loss_fn = nn.MSELoss()
        loss_similarity = loss_fn(cosine_sim, label)
        return loss_similarity


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_model2():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = Identity()
    return model


class ContrastiveLossModel(pl.LightningModule):
    def __init__(self, num_epochs, learning_rate, with_scheduler=True):
        super().__init__()
        self.model = get_model2()

        self.loss_fn = ContrastiveLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.with_scheduler = with_scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        first, second, dis, label = batch
        first_out = self.model(first)
        second_out = self.model(second)
        dis = dis.to(torch.float32)
        loss = self.loss_fn(first_out, second_out, dis)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        first, second, dis, label = batch
        first_out = self.model(first)
        second_out = self.model(second)
        loss = self.loss_fn(first_out, second_out, dis)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.with_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=7, gamma=0.3
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]


def get_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # Update the fully connected layer based on the number of classes in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class LitModel(pl.LightningModule):
    def __init__(self, num_epochs, len_dataset, learning_rate):
        super().__init__()
        self.model = get_model(num_classes=8)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.len_dataset = len_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=self.len_dataset,
        )
        return [optimizer], [scheduler]
