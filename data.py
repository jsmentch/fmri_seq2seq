import torch

def load_datasets(args, train_x=None, valid_x=None, train_y=None, valid_y=None): 

    train_dataset = Dataset(
        split='train',
        random_chunks=False,
        x = train_x,
        y = train_y
    )
    valid_dataset = Dataset(
        split='valid',
        x = valid_x,
        y = valid_y
    )

    return train_dataset, valid_dataset, args


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split='train',
        seq_duration=None,
        random_chunks=False,
        x = None,
        y = None,
    ):
        self.split = split
        self.random_chunks = random_chunks
        self.x = x
        self.y = y
        self.seq_dur = seq_duration
        self.x_len = self.x.shape[0]

        if self.x_len == 0:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):

        if self.random_chunks and self.seq_dur:
            start = random.uniform(0, self.x_len - self.seq_dur)
            end   = start + self.seq_dur
        else:
            start = 0
            end   = self.x_len

        x = self.x[index,start:end]
        y = self.y[index,:]
        # return torch tensors
        return x, y

    def __len__(self):
        return self.x.shape[0]