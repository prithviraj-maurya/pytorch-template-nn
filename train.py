import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import utils

EPOCHS = 25
BATCH_SIZE = 64
PATH = '/kaggle/input/lish-moa/'
num_layers = 3
hidden_size = 256

def run_training():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'    
    df_train = pd.read_csv(PATH + 'train_features.csv')
    targets = pd.read_csv(PATH + 'train_targets_scored.csv')
    utils.get_dummies(df_train, ['cp_type', 'cp_dose', 'cp_time'])
    sig_ids = df_train['sig_id']
    df_train.drop('sig_id', axis=1, inplace=True)
    targets.drop('sig_id', axis=1, inplace=True)

    # TODO use unscored data for training as well
    X_train, X_val, y_train, y_val = train_test_split(
                df_train.values, targets.values, test_size=0.3, random_state=42)

    train_dataset = utils.ModelDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = utils.ModelDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = utils.Model(X_train.shape[1], y_train.shape[1], num_layers, hidden_size)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)    

    engine = utils.Engine(
        model, optimizer, device=DEVICE
    )

    best_loss = np.inf
    early_stopping = 10
    early_stopping_counter = 0

    # TODO use optuns for trails
    for epoch in range(EPOCHS):
        train_loss = engine.train(train_loader)
        val_loss = engine.validate(val_loader)
        scheduler.step(val_loss)

        print(f'Epoch {epoch}, train_loss {train_loss}, val_loss {val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '/models')
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping:
            break
            
    print(f'best loss {best_loss}')
    return best_loss    


loss = run_training()
print(f'Final loss {loss}')