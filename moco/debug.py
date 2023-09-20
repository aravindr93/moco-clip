import builtins
import math
import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import datasets.imagelistdataset, datasets.imagetextdataset
import torchvision.models as models

import moco_clip.loader
import moco_clip.builder

from mjrl.utils.logger import DataLog
from omegaconf import DictConfig, OmegaConf

model = moco_clip.builder.MoCoCLIP(
    vision_encoder=models.__dict__["resnet50"],
    sentence_encoder="distilbert-base-uncased",
    K=100,
    T=0.07,
    mlp=False,
    load_path=None,
)

dataset = datasets.imagetextdataset.RandomImageTextDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
batch = next(iter(dataloader))
model = model.to("cuda:0")
logit, label = model.forward(batch['images'].to("cuda:0"), batch['label'])


# Test the LAION dataset
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.ToTensor(),
        normalize,
])
dataset = datasets.imagetextdataset.LaionDataset(
    location="/datasets01/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..0}/{00055..00055}.tar",
    masks_location='/checkpoint/haideraltahan/laion440m_masks',
    transform=transforms,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
batch = next(iter(dataloader))
logit, label = model.forward(batch["images"].to("cuda:0"), batch["label"])


# Test the ImageNet dataset
dataset = datasets.imagetextdataset.ImageNetDataset(transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
batch = next(iter(dataloader))
logit, label = model.forward(batch["images"].to("cuda:0"), batch["label"])