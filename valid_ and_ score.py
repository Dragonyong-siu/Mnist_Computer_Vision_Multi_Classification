Mnist = torch.load('/content/gdrive/My Drive/' + 'Mnist_Model')

import numpy as np
def Evaluation(dataloader, model, device):
    model.eval()
    Targets = []
    Outputs = []
    with torch.no_grad():
      for bi, Dictionary in enumerate(dataloader):
        x_input = Dictionary['x_input'].to(device)
        target = Dictionary['target'].to(device).squeeze(1)
        Logits = model(x_input)
        output = np.argmax(Logits, axis = 1)
        Targets.extend(target.cpu().detach().numpy().tolist())
        Outputs.extend(output.cpu().detach().numpy().tolist())
    return Targets, Outputs
TARGETS, OUTPUTS = Evaluation(valid_dataloader, Mnist, device)

from sklearn.metrics import accuracy_score
accuracy_score(TARGETS, OUTPUTS)
