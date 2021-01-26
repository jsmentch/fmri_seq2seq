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
    "bs":32,
    "root":"./data",
    "epochs":32,
    'device':'cpu',
    'lr':0.001,
    # Model param
    'input_dim':1,   
    'hidden_dim':256,
    'layer_dim':3,
    'output_dim':9,
    'seq_dim':1024
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
m = model.LSTMClassifier(args['input_dim'], args['hidden_dim'], args['layer_dim'], args['output_dim'], args)
m.to(args['device'])
m = m.float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(m.parameters(), lr=args['lr'], momentum=0.9)

# Train / Valid funcs
def train(args, m, device, train_sampler, optimizer, criterion):

    pbar = tqdm.tqdm(train_sampler)
    for x, y in pbar:
        pbar.set_description("Training batch")
        # zero the parameter gradients
        optimizer.zero_grad()

        x, y = x.to(args['device']), y.to(args['device'])
        # forward + backward + optimize
        y_pred = m(x.float())
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

# # def valid(args, m, device, valid_sampler, writer, epoch):
# #     losses = utils.AverageMeter()
# #     m.eval()
# #     with torch.no_grad():
# #         for x, y in enumerate(valid_sampler):
# #             x = x.to(device)
# #             y = y.to(device)
# #             y_pred = model(x)
# #             loss = torch.nn.functional.mse_loss(y_pred, y)
# #             losses.update(loss.item(), y.size(1))
# #         return losses.avg


# Training stage
t = tqdm.trange(1, args['epochs'] + 1)
for epoch in t:
    t.set_description("Training Epoch")
    end = time.time()
    train_loss = train(args, m, args['device'], train_sampler, optimizer, criterion)