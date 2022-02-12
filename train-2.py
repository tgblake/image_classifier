import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
#import torch.nn.functional as F
#import subprocess
from torch import optim
from PIL import Image
from torchvision import models
from torchvision import datasets, transforms
from collections import OrderedDict
import json
import os
import sys
import argparse

""" train.py:
    Trains a model based on model_arch with a new classifier,
    for epochs iterations, then saves a checkpoint to 
    checkpt_dir/checkpt_file, and shows graph of train_losses 
    and valid_losses.
    
    train_more.py:  Training can be resumed from the checkpoint.
    predict.py:  Accepts an image index and a checkpoint and 
                 predicts the most-probable matches.
    print_log.py  Prints the losses and accuracies from a checkpoint.

    2/10, 2:20   Added a second hidden layer. Dropout 0.3 in both.
    2/10, 3:10   Removed second hidden layer. Dropout 0.5. Learn rate=0.007.
"""


#data_dir_input = sys.argv[1]   # Can't get sys.argv to work, except
#                                 in print_log.  Using parser.

# Basic usage:  cd ImageClassifier
#               python train.py
#          or:  python train.py + parser args

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--data_direc', nargs='?', default='flowers')
parser.add_argument('--model_arch', nargs='?', default='vgg16', help='vgg16, vgg13, or vgg19')
parser.add_argument('--cat_to_name', nargs='?', default='cat_to_name.json')
parser.add_argument('--hidden_units', type=int, nargs='?', default=1024)
parser.add_argument('--checkpt_file', nargs='?', default='checkpt')   # epochs.pth added
parser.add_argument('--learn_rate', type=float, nargs='?', default=0.0008)
parser.add_argument('--epochs', type=int, nargs='?', default=12)
parser.add_argument('--gpu', type=bool, nargs='?', default=True)

inputArgs = parser.parse_args()

print('data_direc = ', inputArgs.data_direc)
print('model_arch = ', inputArgs.model_arch)
print('cat_to_name = ', inputArgs.cat_to_name)
print('hidden_units = ', inputArgs.hidden_units)
print('checkpt_file = ', inputArgs.checkpt_file)
print('learn_rate = ', inputArgs.learn_rate)
print('epochs = ', inputArgs.epochs)
print('gpu = ', inputArgs.gpu, '\n')


device = torch.device("cuda" if inputArgs.gpu and torch.cuda.is_available() else "cpu")


data_dir = inputArgs.data_direc   #'flowers'  'ImageClassifier/' + 
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224), 
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the transforms, define the dataloaders
#dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


with open(inputArgs.cat_to_name, 'r') as f:
    cat_to_name = json.load(f)
    
# For printing indices and flower names:
cat_to_name_sorted = sorted(cat_to_name.items())   #, key=int)
cat_to_name_ord = OrderedDict(cat_to_name_sorted)
print(cat_to_name_ord)   #.items())


dataiter = iter(train_loader)
images, labels = dataiter.next()
print('type(images) = ', type(images))
print('images.shape = ', images.shape)
print('labels.shape = ', labels.shape)
print('images[1].shape = ', images[1].shape)
print('images[1]..squeeze.shape = ', images[1].numpy().squeeze().shape)
print('\n')

# added a second hidden layer:
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, inputArgs.hidden_units)),   #1024
                                        ('relu', nn.ReLU()),
                                        ('Dropout', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(inputArgs.hidden_units, 102)),
                                        #('relu', nn.ReLU()),
                                        #('Dropout', nn.Dropout(0.3)),
                                        #('fc3', nn.Linear(384, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))

if inputArgs.model_arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif inputArgs.model_arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif inputArgs.model_arch == 'vgg19':
    model = models.vgg19(pretrained=True)
#elif inputArgs.model_arch == 'resnet50':
#    model = models.resnet50(pretrained=True)
else:
    print('model_arch not one of accepted ones:')
    print('    vgg16, vgg13 or vgg19 \n')
    
for name, param in model.named_parameters():
    param.requires_grad=False

model.classifier = classifier
print('model_arch  = ', inputArgs.model_arch)
print(model.classifier)
print('\n')


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=inputArgs.learn_rate)   # 0.001
LRscheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

model.to(device)
    
#epochs = 16

train_losses, valid_losses, accuracies, learn_rates = [], [], [], []

for e in range(inputArgs.epochs):
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                #images = images.view(images.shape[0], -1)
                log_ps = model(images)
                valid_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        curr_lr = optimizer.param_groups[0]['lr']
        #print(curr_lr)
        model.train()
        
        train_losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        accuracies.append(accuracy / len(valid_loader))
        learn_rates.append(curr_lr)
        
        print('Epoch: {}/{} .. '.format(e+1, inputArgs.epochs),
              'Training loss: {:.3f} .. '.format(running_loss / len(train_loader)),
              'Validation loss: {:.3f} .. '.format(valid_loss / len(valid_loader)),
              'Validation accuracy: {:.2f}'.format(100 * accuracy / len(valid_loader)),
              'LR: {:.4e}'.format(curr_lr))
        LRscheduler.step()


checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer': optimizer,   #_state_dict(),
              'epoch': inputArgs.epochs,
              'class_to_index': cat_to_name,
              'criterion': criterion,
              'train_losses': train_losses,
              'valid_losses': valid_losses,
              'accuracies': accuracies,
              'LRscheduler': LRscheduler,
              'learn_rates': learn_rates,
              'train_loader': train_loader,
              'valid_loader': valid_loader}


checkpt_file = 'checkpt_dir/' + inputArgs.checkpt_file + '_'  # + inputArgs.model_arch + '_'
checkpt_file = checkpt_file + str(inputArgs.epochs) +'.pth'
print('\nCheckpoint:  ', checkpt_file, '\n')

torch.save(checkpoint, checkpt_file)
