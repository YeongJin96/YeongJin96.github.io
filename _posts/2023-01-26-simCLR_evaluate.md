---
layout: post
title: SimCLR_evaluate code
date: 2023-01-26 19:09:00
description: SimCLR 모델을 평가할 때 사용한 코드입니다.
tags: SimCLR
categories: Experiment
---

```python
import torch
import torchvision
import numpy as np
import argparse
import torch.nn as nn
```


```python
use_tpu = False
if use_tpu:
  VERSION = "20200220" #@param ["20200220","nightly", "xrt==1.15.0"]
  !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
  !python pytorch-xla-env-setup.py --version $VERSION
```


```python
class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)
```


```python
def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch
```


```python
def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


```


```python
class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
```


```python
import os
import yaml

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f) or {}
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg
```


```python
def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
```


```python
class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
```


```python
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
```


```python
from pprint import pprint

parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("./config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args([])

if use_tpu:
  args.device = dev
else:
  args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


```python
args.batch_size = 128
args.dataset = "Custom"
args.resnet = "resnet18"
args.model_path = "../save_models"
args.epoch_num = 100
logistic_batch_size = 128
args.logistic_epochs = 500
```


```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root='../data/PNU_x40', transform=TransformsSimCLR(size=args.image_size).test_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=args.logistic_batch_size,
                                           drop_last=True,
                                           shuffle = True,
                                           num_workers=args.workers,)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=args.logistic_batch_size,
                                          drop_last=True,
                                          shuffle = False,
                                          num_workers=args.workers,)

```


```python
encoder = get_resnet(args.resnet, pretrained=False) # don't load a pre-trained model from PyTorch repo
n_features = encoder.fc.in_features  # get dimensions of fc layer

# load pre-trained model from checkpoint
simclr_model = SimCLR(encoder, args.projection_dim, n_features)
model_fp = os.path.join(
    args.model_path, "checkpoint50_500.tar"
)
simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
simclr_model = simclr_model.to(args.device)
```



```python
## Logistic Regression
n_classes = 3
model = LogisticRegression(simclr_model.n_features, n_classes)
model = model.to(args.device)
```


```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
```


```python
def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(context_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, context_model, device)
    test_X, test_y = inference(test_loader, context_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
```


```python
print("### Creating features from pre-trained context model ###")
(train_X, train_y, test_X, test_y) = get_features(
    simclr_model, train_loader, test_loader, args.device
)

arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
    train_X, train_y, test_X, test_y, args.logistic_batch_size
)
```

    ### Creating features from pre-trained context model ###
    Step [0/59]	 Computing features...
    Step [20/59]	 Computing features...
    Step [40/59]	 Computing features...
    Features shape (15104, 512)
    Step [0/14]	 Computing features...
    Features shape (3584, 512)



```python
for epoch in range(args.logistic_epochs):
    loss_epoch, accuracy_epoch = train(args, arr_train_loader, simclr_model, model, criterion, optimizer)
    
    if epoch % 10 == 0:
      print(f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")


# final testing
loss_epoch, accuracy_epoch = test(
    args, arr_test_loader, simclr_model, model, criterion, optimizer
)
print(
    f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
)
```

    Epoch [0/500]	 Loss: 0.5930819640818396	 Accuracy: 0.7464535361842105
    Epoch [10/500]	 Loss: 0.39237397085679204	 Accuracy: 0.8347553453947368
    Epoch [20/500]	 Loss: 0.3644816267647241	 Accuracy: 0.847399259868421
    Epoch [30/500]	 Loss: 0.3481225881137346	 Accuracy: 0.8536698190789473
    Epoch [40/500]	 Loss: 0.3363633563644008	 Accuracy: 0.8597347861842105
    Epoch [50/500]	 Loss: 0.3270935043692589	 Accuracy: 0.8640522203947368
    Epoch [60/500]	 Loss: 0.31941156481441696	 Accuracy: 0.8676500822368421
    Epoch [70/500]	 Loss: 0.3128436703823115	 Accuracy: 0.8699629934210527
    Epoch [80/500]	 Loss: 0.30710559750073835	 Accuracy: 0.8726356907894737
    Epoch [90/500]	 Loss: 0.3020114884956887	 Accuracy: 0.8745888157894737
    Epoch [100/500]	 Loss: 0.2974322961741372	 Accuracy: 0.8768503289473685
    Epoch [110/500]	 Loss: 0.29327447614387464	 Accuracy: 0.8784436677631579
    '
    '
    '
    Epoch [400/500]	 Loss: 0.23650363087654114	 Accuracy: 0.9050678453947368
    Epoch [410/500]	 Loss: 0.23541850380991636	 Accuracy: 0.9056332236842105
    Epoch [420/500]	 Loss: 0.23435954435875542	 Accuracy: 0.9058388157894737
    Epoch [430/500]	 Loss: 0.2333255169030867	 Accuracy: 0.9061472039473685
    Epoch [440/500]	 Loss: 0.232315268171461	 Accuracy: 0.9066097861842105
    Epoch [450/500]	 Loss: 0.2313277190060992	 Accuracy: 0.9071751644736842
    Epoch [460/500]	 Loss: 0.2303618745584237	 Accuracy: 0.9076377467105263
    Epoch [470/500]	 Loss: 0.22941679507493973	 Accuracy: 0.908203125
    Epoch [480/500]	 Loss: 0.2284916174647055	 Accuracy: 0.9087685032894737
    Epoch [490/500]	 Loss: 0.22758550843910166	 Accuracy: 0.9093852796052632
    [FINAL]	 Loss: 0.3037595466563576	 Accuracy: 0.8743832236842105



```python

```
