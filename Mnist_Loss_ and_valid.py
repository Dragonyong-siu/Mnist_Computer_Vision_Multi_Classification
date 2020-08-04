def Mnist_Loss(Logits, target):
  loss_function = nn.CrossEntropyLoss()
  Loss = loss_function(Logits, target) 
  return Loss
