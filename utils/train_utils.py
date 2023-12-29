import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_classify(model:torch.nn.Module, dataloader:DataLoader, criterion:nn.Module, device:str=device) -> (float, float):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_hat = model(X)
            loss = criterion(y_hat, y)

            total_loss += loss.item()
            total_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

    model.train()
    
    return total_loss / total, total_correct / total * 100


def train_classify(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, 
                   optimizer:torch.optim.Optimizer, criterion:nn.Module, num_epochs:int=10, device:str=device) -> None:
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            y_hat = model(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
        
            progress_bar.set_description(f"Epoch {epoch+1}")
            progress_bar.set_postfix(loss=total_loss/(i+1), accuracy=100.*total_correct/total)
        
        print(f"Epoch: {epoch + 1}, loss: {total_loss / len(train_loader)}, acc: {100. * total_correct / total}")
        test_loss, test_acc = evaluate_classify(model, test_loader, criterion)
        print(f"Epoch: {epoch + 1}, test loss: {test_loss}, test acc: {test_acc}")