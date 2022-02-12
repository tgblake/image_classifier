import numpy as np
import torch
#import matplotlib.pyplot as plt
from torch import nn
#import torch.nn.functional as F
#import subprocess
from torch import optim
from PIL import Image
from torchvision import models
from collections import OrderedDict
import os
import sys
import argparse
from load_checkpoint import load_checkpoint

# Basic usage:  python train_more.py --checkpt_in checkpt2.pth
parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--checkpt_in', default='checkpt2', help='Omit folder and .pth')
#parser.add_argument('--checkpt_out', default='checkpt')   # +[total epochs].pth added
parser.add_argument('--add_epochs', type=int, default=4)
parser.add_argument('--gpu', default=True)

inputArgs = parser.parse_args()

checkpt_in_file = 'checkpt_dir/' + inputArgs.checkpt_in + '.pth'
print('checkpt_in_file = ', checkpt_in_file)
#print('checkpt_out = ', inputArgs.checkpt_out)

device = torch.device("cuda" if inputArgs.gpu and torch.cuda.is_available() else "cpu")
#print('device = ', device)


model, epoch, optimizer, criterion, train_losses, valid_losses, accuracies, LRscheduler, learn_rates, train_loader, valid_loader =                load_checkpoint(checkpt_in_file)

model.to(device)

#print(model.state_dict().keys())
#print(optimizer.state_dict().keys())
print('Starting epoch: ', epoch, '\n') 
#print(model.classifier)


# Runnng trainer more, after loadiing checkpoint:

epochs = epoch + inputArgs.add_epochs   

for e in range(epoch, epochs):
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        #images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)  #model.forward(images)
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
        model.train()
        curr_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        accuracies.append(accuracy / len(valid_loader))
        learn_rates.append(curr_lr)
        
        print('Epoch: {}/{} .. '.format(e+1, epochs),
              'Training loss: {:.3f} .. '.format(running_loss / len(train_loader)),
              'Validation loss: {:.3f} .. '.format(valid_loss / len(valid_loader)),
              'Validation accuracy: {:.2f} .. '.format(100 * accuracy / len(valid_loader)),
              'LR: {:.4e}'.format(curr_lr))
        LRscheduler.step()
 

checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer': optimizer,   #_state_dict(),
              'epoch': epochs,
              'class_to_index': model.class_to_idx,
              'criterion': criterion,
              'train_losses': train_losses,
              'valid_losses': valid_losses,
              'accuracies': accuracies,
              'LRscheduler': LRscheduler,
              'learn_rates': learn_rates,
              'train_loader': train_loader,
              'valid_loader': valid_loader}

checkpt_file = 'checkpt_dir/' + inputArgs.checkpt_in + '_' + str(epochs) + '.pth'
print('\nSaving checkpoint to ', checkpt_file, '\n')
torch.save(checkpoint, checkpt_file)
              
        
#plt.plot(train_losses, label='Training loss')
#plt.plot(valid_losses, label='Validation loss')
#plt.plot(accuracies, label='Validation accuracy')
#plt.legend(frameon=False)

             
                    