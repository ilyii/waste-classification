import torch

def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()