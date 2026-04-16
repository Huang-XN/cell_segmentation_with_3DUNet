
from torch import nn 
import torch.optim as optim
import torch

from torch.amp import autocast, GradScaler

def train(train_loader,
          model,
          optimizer,
          num_epoch=10,
          print_every=100):
    """
    Train a model on CIFAR-10 using ResNet built from PyTorch

    Inputs:
    - dataset: Dict containing training and validation dataset.
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model.
    - epochs: (Optional) A Python integer giving the number of epochs to train for.
    - device: A string indicating training with 'cpu' or 'cuda'

    Returns: Nothing, but prints model accuracies during training.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Trainig model on {device}')

    model = model.to(device)

    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(num_epoch):
        print(f"{ep} Epoch")
        for t, (x,y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)

            optimizer.zero_grad()

            with autocast('cuda'):
                scores = model(x)
                loss = criterion(scores, y)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if t % print_every == 0:
                print("Iteration %d, loss = %.4f" % (t, loss.item()))
                #check_acc(val_loader, model)
                print()
            