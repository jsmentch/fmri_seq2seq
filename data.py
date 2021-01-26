import torch
import random

def load_datasets(args, train_x=None, valid_x=None, train_y=None, valid_y=None): 

    train_dataset = Dataset(
        split='train',
        random_chunks=False,
        seq_duration = args['seq_dim'],
        input_dim = args['input_dim'],
        x = train_x,
        y = train_y
    )
    valid_dataset = Dataset(
        split='valid',
        random_chunks=False,
        seq_duration = args['seq_dim'],
        input_dim = args['input_dim'],
        x = valid_x,
        y = valid_y
    )

    return train_dataset, valid_dataset, args


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split='train',
        seq_duration=None,
        input_dim = None,
        random_chunks=False,
        x = None,
        y = None,
    ):
        self.split = split
        self.random_chunks = random_chunks
        self.x = x
        self.y = y
        self.seq_dur = seq_duration
        self.input_dim = input_dim
        self.shape = [seq_duration,input_dim]
        self.x_len = self.x.shape[0]

        if self.x_len == 0:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):

        start = int(random.uniform(0, self.x.shape[-1] - self.seq_dur))
        end   = int(start + self.seq_dur)

        x = self.x[index,start:end]
        y = self.y[index,:]

        return x[:,None], y

    def __len__(self):
        return self.x.shape[0]