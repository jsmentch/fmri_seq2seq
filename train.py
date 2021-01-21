import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import model
import data
from sklearn.model_selection import train_test_split
import tqdm
import time

args = {
    "bs":16,
    "root":"./data",
    "epochs":32,
    'device':'cpu',
    'lr':0.001,
    'dur':1000
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

# Model
m = model.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(m.parameters(), lr=args['lr'], momentum=0.9)

# Train / Valid funcs
def train(args, m, device, train_sampler, optimizer, criterion):

    for x, y in train_sampler:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        print("x:"+str(x.shape))
        print("y:"+str(y.shape))
        y_pred = m(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

def valid(args, m, device, valid_sampler, writer, epoch):
    losses = utils.AverageMeter()
    m.eval()
    with torch.no_grad():
        for x, y in enumerate(valid_sampler):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            losses.update(loss.item(), y.size(1))
        return losses.avg


# Training stage
t = tqdm.trange(1, args['epochs'] + 1)
for epoch in t:
    t.set_description("Training Epoch")
    end = time.time()
    train_loss = train(args, m, args['device'], train_sampler, optimizer, criterion)