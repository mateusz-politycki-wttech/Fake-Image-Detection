import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50
from networks.vit import ViTModel
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--file', default='examples_realfakedir')
parser.add_argument('-m','--model_path', type=str, default='checkpoints/blur_jpg_prob0.5/model_epoch_best.pth')
# parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-c','--crop', type=int, default=224, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')

opt = parser.parse_args()

# model = resnet50(num_classes=1)
model = ViTModel()
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])

torch.save(model.state_dict(), "model.pt")

if(not opt.use_cpu):
  model.cuda()
model.eval()

# Transform
def custom_resize(img, opt):
    interp = 'bilinear'
    return TF.resize(img, (224,224), interpolation=Image.BILINEAR)
  
# trans_init =  [transforms.Lambda(lambda img: custom_resize(img, opt))]
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
  
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = trans(Image.open(opt.file).convert('RGB'))

with torch.no_grad():
    in_tens = img.unsqueeze(0)
    print(in_tens.shape)
    if(not opt.use_cpu):
    	in_tens = in_tens.cuda()
    prob = model(in_tens).sigmoid().item()

print('probability of being synthetic: {:.2f}%'.format(prob * 100))
