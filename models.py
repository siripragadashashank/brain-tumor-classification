import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from efficientnet_pytorch_3d import EfficientNet3D


class EfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.net(x)
        return out


class SimpleCNNModel(nn.Module):
    def __init__(self, width=128, height=128, depth=64):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=depth, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.bn4 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.bn4 = nn.BatchNorm3d(256)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = f.relu(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = f.relu(x)
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time, train_auc, train_acc, train_f1score = self.train_epoch(train_loader)
            valid_loss, valid_time, valid_auc, val_acc, val_f1score = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f}s, auc: {:.4f}, acc: {:.4f}, f1: {:.4f}",
                n_epoch, train_loss, train_time, train_auc, train_acc, train_f1score
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, time: {:.2f}s, auc: {:.4f}, acc: {:.4f}, f1: {:.4f}",
                n_epoch, valid_loss, valid_time, valid_auc, val_acc, val_f1score
            )

            # if True:
            # if self.best_valid_score < valid_auc:
            if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                    "loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0
        y_true = []
        y_hat = []

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            y_true.extend(batch["y"].tolist())
            y_hat.extend(torch.sigmoid(outputs).tolist())

            self.optimizer.step()
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss / step, end="\r")

        y_true = [1 if x > 0.5 else 0 for x in y_true]
        train_auc = roc_auc_score(y_true, y_hat)
        y_hat = [1 if x > 0.5 else 0 for x in y_hat]
        train_acc = accuracy_score(y_true, y_hat)
        train_f1score = f1_score(y_true, y_hat)

        return sum_loss / len(train_loader), int(time.time() - t), train_auc, train_acc, train_f1score

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch["y"].tolist())
                outputs_all.extend(torch.sigmoid(outputs).tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        y_hat = [1 if x > 0.5 else 0 for x in outputs_all]
        acc = accuracy_score(y_all, y_hat)
        f1score = f1_score(y_all, y_hat)

        return sum_loss / len(valid_loader), int(time.time() - t), auc, acc, f1score

    def save_model(self, n_epoch, save_path, loss, auc):
        self.lastmodel = f"{save_path}-e{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


