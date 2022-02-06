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
import argparse
from util_functions import load_data, create_model, save_checkpoint

parser = argparse.ArgumentParser(description="Train the Image Classifier")
parser.add_argument('data_dir', default='flowers', help='Data Directory', type=str)
parser.add_argument('--save_dir', default='.', help='Save Directory', type=str)
parser.add_argument('--arch', default='vgg16', help='Choose VGG16 or DenseNet121 Architecture (Default: VGG16)', type=str)
parser.add_argument('--learning_rate', default=0.001, help='Learning Rate (Default: 0.001)', type=float)
parser.add_argument('--hidden_units', help='Hidden Units (Default: 2048 for VGG16 and 256 for DenseNet121)', type=int)
parser.add_argument('--epochs', default=5, help='Number of Epochs (Default: 5)', type=int)
parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
args = parser.parse_args()

# Load Data
train_data, trainloader, validloader = load_data(args.data_dir)
# Check GPU
device = torch.device("cuda" if args.gpu else "cpu")
# Create Model
model, criterion, optimizer = create_model(arch=args.arch, hidden_units=args.hidden_units, learning_rate=args.learning_rate)
# Train and Print Results
epochs = args.epochs
steps = 0
print_every = 5
model = model.to(device)
with active_session():
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss += criterion(logps, labels).item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += equals.type(torch.FloatTensor).mean()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Training Finished")
# Save Model
save_checkpoint(model=model, path=args.save_dir, idx=train_data, optim=optimizer, crit=criterion, e=epochs)
