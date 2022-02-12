import torch
from torchvision import models


def load_checkpoint(path):
    """ Called from other .py files.  Must have GPU active for all.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('load_checkpoint:  device = ', device)
    #if device == "cuda":
    #    print('checkpoint = torch.load(path, map_location="cuda")')
    #    checkpoint = torch.load(path, map_location="cuda")
    #else:
    #    print('checkpoint = torch.load(path, map_location="cpu")')
    #    checkpoint = torch.load(path, map_location="cpu")
    
    checkpoint = torch.load(path)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_index']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    accuracies = checkpoint['accuracies']
    LRscheduler = checkpoint['LRscheduler']
    learn_rates = checkpoint['learn_rates']
    train_loader = checkpoint['train_loader']
    valid_loader = checkpoint['valid_loader']
    
    return model, epoch, optimizer, criterion, train_losses, valid_losses, accuracies, LRscheduler, learn_rates, train_loader, valid_loader

