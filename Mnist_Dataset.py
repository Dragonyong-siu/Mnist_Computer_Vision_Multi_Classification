import torch
class Mnist_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    Dictionary = {}
    x_input = self.data.iloc[idx, 1:]
    x_input = torch.Tensor(x_input)
    target = self.data.iloc[idx, 0]
    target = torch.Tensor([target]).long()
    Dictionary['x_input'] = x_input
    Dictionary['target'] = target
    return Dictionary

from torch.utils.data import DataLoader
BATCH_SIZE = 16
train_dataloader = DataLoader(Mnist_Dataset(mnist_data), batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
