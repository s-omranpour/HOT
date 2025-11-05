import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler

class TSDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path,
        name,
        split_sizes=[0.7, 0.1, 0.2],
        context_length=96, 
        prediction_length=96, 
        normalize=True,
        batch_size=16, 
        num_workers=2
    ):
        super().__init__()
        assert sum(split_sizes) == 1.

        self.data_path = data_path
        self.name = name
        self.split_sizes = split_sizes
        self.batch_size = batch_size
        self.context_length = context_length
        self.normalize = normalize
        self.num_workers = num_workers
        self.prediction_length = prediction_length
        self.scaler = StandardScaler()
        self.datasets = self.init()
        

    def init(self):
        ts = np.load(f'{self.data_path}/{self.name}/{self.name}.npy')
        self.n_vars = ts.shape[0]
        self.n_timesteps = ts.shape[1]
        train_size = int(self.split_sizes[0] * self.n_timesteps)
        val_size = int(self.split_sizes[1] * self.n_timesteps)
        test_size = self.n_timesteps - val_size - train_size

        if self.normalize:
            self.scaler.fit(ts[:, :train_size].T)
            ts = self.scaler.transform(ts.T).T

        datasets = {}
        datasets['train'] = MultivarTSDataset(
            data=ts[:, :train_size], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        datasets['val'] = MultivarTSDataset(
            data=ts[:, train_size - self.context_length : train_size + val_size], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        
        datasets['test'] = MultivarTSDataset(
            data=ts[:, - test_size - self.context_length:], 
            context_length=self.context_length, 
            prediction_length=self.prediction_length
        )
        return datasets
        

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )


class MultivarTSDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        super().__init__()
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_vars = data.shape[0]
        self.n_timesteps = data.shape[1]

    def __getitem__(self, index):
        x = torch.tensor(
            self.data[:, index:index + self.context_length]
        )
        y = torch.tensor(
            self.data[:, index + self.context_length : index + self.context_length + self.prediction_length]
        )
        return x.float(), y.float()


    def __len__(self):
        return self.data.shape[1] - self.context_length - self.prediction_length
