# reference: https://www.kaggle.com/artgor/code-for-live-pair-coding, https://www.youtube.com/watch?v=VRVit0-0AXE
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class ModelDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx, :], dtype=torch.float),
            'targets': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['features'].to(self.device)
            targets = data['targets'].to(self.device)
            output = self.model(inputs)
            loss = self.loss(output, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)  

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data['features'].to(self.device)
            targets = data['targets'].to(self.device)
            output = self.model(inputs)
            loss = self.loss(output, targets)
            final_loss += loss.item()
        return final_loss / len(data_loader)     

class Model(nn.Module):
    def __init__(self, num_features, num_targets, num_layers, hidden_size, dropout=0.2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, num_targets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

def get_dummies(df, columns):
    ohe = pd.get_dummies(df[columns])
    ohe_columns = [f"{columns}_{c}" for c in ohe.columns]
    ohe.columns = ohe_columns
    df.drop(columns, axis=1, inplace=True)
    df = df.join(ohe)
    return df