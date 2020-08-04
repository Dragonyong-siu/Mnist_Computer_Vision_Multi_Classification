device = 'cpu'
import torch.nn.functional as F
import torch.nn as nn
class Mnist_Model(nn.Module):
  def __init__(self):
    super(Mnist_Model, self).__init__()
    self.linear1 = nn.Linear(784, 392)
    self.linear2 = nn.Linear(392, 196)
    self.linear3 = nn.Linear(196, 98)
    self.linear4 = nn.Linear(98, 49)
    self.linear5 = nn.Linear(49, 10)
    self.dropout = nn.Dropout(0.3)
  def forward(self, x_input):
    x_input = self.linear1(x_input)
    x_input = F.relu(x_input)
    x_input = self.linear2(x_input)
    x_input = F.relu(x_input)
    x_input = self.linear3(x_input)
    x_input = F.relu(x_input)
    x_input = self.linear4(x_input)
    x_input = F.relu(x_input)
    x_input = self.linear5(x_input)
    Logits = self.dropout(x_input)
    Logits_softmax = torch.softmax(Logits, dim = 0)
    
    return Logits_softmax

Mnist = Mnist_Model().to(device)
