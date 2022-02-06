import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from workspace_utils import active_session
from collections import OrderedDict
from PIL import Image

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return train_data, trainloader, validloader

def label_map(file):
    with open(file, 'r') as f:
        label_map = json.load(f)
    return label_map

def create_model(arch, hidden_units, learning_rate):
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            model.classifier = nn.Sequential(OrderedDict ([
                                            ('fc1', nn.Linear(25088, 4096)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(0.3)),
                                            ('fc2', nn.Linear(4096, hidden_units)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout(0.3)),
                                            ('fc3', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
        else:
            model.classifier = nn.Sequential(OrderedDict ([
                                        ('fc1', nn.Linear(25088, 4096)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(0.3)),
                                        ('fc2', nn.Linear(4096, 2048)),
                                        ('relu2', nn.ReLU()),
                                        ('dropout2', nn.Dropout(0.3)),
                                        ('fc3', nn.Linear(2048, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))   
    else:
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            model.classifier = nn.Sequential(OrderedDict ([
                                        ('fc1', nn.Linear(1024, hidden_units)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(0.3)),
                                        ('fc2', nn.Linear(hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
        else:
            model.classifier = nn.Sequential(OrderedDict ([
                                        ('fc1', nn.Linear(1024, 256)), #set default number
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(0.3)),
                                        ('fc2', nn.Linear(256, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    criterion = nn.NLLLoss()
    if learning_rate:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    return model, criterion, optimizer

def save_checkpoint(model, path, idx, optim, crit, e):
    model.class_to_idx = idx.class_to_idx
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optim.state_dict,
                  'criterion': crit,
                  'epochs': e,
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, path+'/checkpoint.pth')

def load_checkpoint(filepath, gpu):    
    if gpu:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array '''
    img = Image.open(path)
    width, height = img.size
    aspect_ratio = width / height 
    if aspect_ratio > 1: # width greater than height
        img = img.resize((round(aspect_ratio*256), 256))
    else:
        img = img.resize((256, round(256/aspect_ratio)))
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((round(left), round(top), round(right), round(bottom)))
    np_img = np.array(img) / 225
    np_img -= np.array([0.485, 0.456, 0.406])
    np_img /= np.array([0.229, 0.224, 0.225])
    np_img = np_img.transpose((2,0,1))
    return np_img