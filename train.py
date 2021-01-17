import os
import torch
import numpy as np

import model
import data
from sklearn.model_selection import train_test_split

args = {
    "bs":16,
    "root":"./data"
}

# Data
x = np.load(os.path.join(args['root'],'brain_train.npy'))
y = np.load(os.path.join(args['root'],'char_train.npy'))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)
train_dataset, valid_dataset, args = data.load_datasets(args, train_x=x_train, valid_x=x_val, train_y=y_train, valid_y=y_val)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['bs'], shuffle=True, drop_last=True
)

valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1, drop_last=True
)