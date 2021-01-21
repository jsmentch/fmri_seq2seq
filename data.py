import torch

def load_datasets(args, train_x=None, valid_x=None, train_y=None, valid_y=None): 

    train_dataset = Dataset(
        split='train',
        random_chunks=False,
        seq_duration = args['dur'],
        x = train_x,
        y = train_y
    )
    valid_dataset = Dataset(
        split='valid',
        seq_duration = args['dur'],
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
        self.shape = [300,300]
        self.x_len = self.x.shape[0]

        if self.x_len == 0:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):

        start = 0
        end   = (self.shape[0]*self.shape[1])

        x = self.x[index,start:end]
        x = torch.reshape(x,(x.size()[0],self.shape[0],self.shape[1]))
        y = self.y[index,:]

        return x, y

    def __len__(self):
        return self.x.shape[0]