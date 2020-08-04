from tqdm import tqdm
def Train_def(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0

  for bi, Dictionary in enumerate(Book):
    x_input = Dictionary['x_input'].to(device)
    target = Dictionary['target'].to(device).squeeze(1)
    model.zero_grad()
    Logits_softmax = model(x_input)
    Loss = Mnist_Loss(Logits_softmax, target)
    Loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += Loss.item()
  average_train_loss = total_loss / len(dataloader)
  print(" Average training loss: {0:.2f}".format(average_train_loss))  
  

lr = 0.0002
def Fit(dataloader, EPOCHS = 20):
  optimizer = torch.optim.AdamW(Mnist.parameters(), lr = lr)
   
  for i in range(EPOCHS):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_def(train_dataloader, Mnist, optimizer, device)    
  torch.save(Mnist, '/content/gdrive/My Drive/' + f'Mnist_Model')
    
Fit(train_dataloader)
