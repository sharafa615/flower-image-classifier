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
from util_functions import label_map, load_checkpoint, process_image

parser = argparse.ArgumentParser(description="Predict Using the Image Classifier")
parser.add_argument('input', help='Path of Input Image', type=str)
parser.add_argument('checkpoint', default='checkpoint.pth', help='Path of Model Checkpoint', type=str)
parser.add_argument('--top_k', default=1, help='Number of Top Classes to Display (Default: 1)', type=int)
parser.add_argument('--category_names', default='cat_to_name.json', help='Map of Categories to Real Names', type=str)
parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
args = parser.parse_args()

# Map Labels
cat_to_name = label_map(args.category_names)
# Load Checkpoint
model = load_checkpoint(args.checkpoint, args.gpu)
# Check GPU
device = torch.device("cuda" if args.gpu else "cpu")
# Predict
image = process_image(args.input)
model.to(device)
model.eval()
with torch.no_grad():
    image = torch.from_numpy(image)
    image = image.unsqueeze(dim=0)
    image = image.type(torch.FloatTensor)
    image = image.to(device)
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_ps, top_classes = ps.topk(args.top_k, dim=1)
    top_ps = top_ps.cpu()
    top_classes = top_classes.cpu()
    top_ps = top_ps.numpy()
    list_ps = [float(p) for p in top_ps[0]]
    mapping = {val:key for key, val in model.class_to_idx.items()}
    list_class = [mapping[int(idx)] for idx in top_classes[0]]
top_flowers = [cat_to_name[idx] for idx in list_class]
print("Top K Predictions")
print("{:5}   {:11}   {}".format('Class', 'Probability', 'Flower'))
for c, ps, flower in zip(list_class, list_ps, top_flowers):
    print(f"{c:>5}   {ps:11.5f}   {flower}")
