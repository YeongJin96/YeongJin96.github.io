---
layout: post
title: SimCLR_train code
date: 2023-01-26 19:09:00
description: SimCLR 모델을 학습할 때 사용한 코드입니다.
tags: SimCLR
categories: code
---

```python
import torch
import torchvision
import torch.nn as nn
import os
```

torchvision의 transforms을 사용해서, 이미지 Augmentation을 생성하는 클래스입니다.

train_transform은 train용 augmentation으로, resize&crop, flip, color_jitter를 사용했고, 0.2의 확률로 grayscale로 변환한 뒤, 텐서로 만들고, SimCLR은 1개의 이미지당 2개의 augmentation 이미지를 필요로 하므로, 같은 transform으로 두개의 이미지를 return합니다.


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



Github에는 yaml파일로 parameters를 불러오는 방식을 사용했지만, yaml파일 없이 코드실행을 위해 따로 parameters를 정의했습니다.


```python
from torchvision.datasets import ImageFolder
input_size = 224
batch_size = 128
weight_decay = 1e-06
epochs = 500
model_path = "../save_models"
current_epoch = 0
temperature = 0.5
```



Dataset정의

가장 위에 정의했던 transfrom클래스를 사용해 이미지들을 augmentation후, data_loader를 만듭니다.


```python
train_dataset = ImageFolder(root='../data/PNU_all', transform=TransformsSimCLR(size=input_size))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=8,
                                           drop_last=True)
```



pytorch에 정의되어있는 resnet18과 resnet50을 입력에 따라 불러옵니다.


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



입력으로 받은것을 그대로 return해주는 Identity클래스 입니다. (ResNet의 skip connection)


```python
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
```



LARS Optimizer입니다. (설명 링크)


```python
"""
LARS: Layer-wise Adaptive Rate Scaling
Converted from TensorFlow to PyTorch
https://github.com/google-research/simclr/blob/master/lars_optimizer.py
"""

from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(
                            g_norm.ge(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True
```



SimCLR 모델입니다. 

encoder는 feature를 추출하기 위한 CNN based 모델입니다.

n_features는 MLP(projector)레이어의 parameter로 사용되었습니다.

projectrion_dim은 최종적으로 추출할 feature의 갯수입니다.


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
def load_optimizer(optimizer, model):

    scheduler = None
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(model_path, current_epoch, model, optimizer):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
```


```python
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
```



Augmentation된 이미지 쌍(pair)의 similarity계산을 위한 NT_Xent loss함수 입니다.


```python
import torch.distributed as dist

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
```


```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
```


```python
def train(global_step, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), global_step)
        loss_epoch += loss.item()
        global_step += 1
    return loss_epoch
```


```python
resnet = "resnet50"
encoder = get_resnet(resnet, pretrained=False)
n_features = encoder.fc.in_features  # get dimensions of fc layer

projection_dim = 64
model = SimCLR(encoder, projection_dim, n_features)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

optimizer, scheduler = load_optimizer("LARS", model)
```

```python
criterion = NT_Xent(batch_size, temperature, world_size=1)
```


```python
global_step = 0
current_epoch = 0
start_epoch = 0
for epoch in range(start_epoch, epochs):
    lr = optimizer.param_groups[0]["lr"]
    loss_epoch = train(global_step, train_loader, model, criterion, optimizer, writer)
    
    if scheduler:
        scheduler.step()
        
    if epoch % 10 == 0:
        save_model(model_path, current_epoch, model, optimizer)
        
    writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
    writer.add_scalar("Misc/learning_rate", lr, epoch)
    print(
        f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
    )
    current_epoch += 1
    
save_model(model_path, current_epoch, model, optimizer)
```


    Step [0/192]	 Loss: 5.547216415405273
    Step [50/192]	 Loss: 5.5168609619140625
    Step [100/192]	 Loss: 5.466494083404541
    Step [150/192]	 Loss: 5.5113701820373535
    Epoch [0/500]	 Loss: 5.47970977673928	 lr: 0.15
    Step [0/192]	 Loss: 5.416123390197754
    Step [50/192]	 Loss: 5.365357398986816
    Step [100/192]	 Loss: 5.415189266204834
    Step [150/192]	 Loss: 5.289378643035889
    Epoch [1/500]	 Loss: 5.405081287026405	 lr: 0.15
    Step [0/192]	 Loss: 5.412885665893555
    Step [50/192]	 Loss: 5.288458347320557
    Step [100/192]	 Loss: 5.321774005889893
    Step [150/192]	 Loss: 5.202507972717285
    Epoch [2/500]	 Loss: 5.278357977668445	 lr: 0.14999
    Step [0/192]	 Loss: 5.205842971801758
    Step [50/192]	 Loss: 5.146916389465332
    Step [100/192]	 Loss: 5.040021896362305
    Step [150/192]	 Loss: 5.046496391296387
    Epoch [3/500]	 Loss: 5.109636095662911	 lr: 0.14999
    Step [0/192]	 Loss: 5.132671356201172
    Step [50/192]	 Loss: 4.919950008392334
    Step [100/192]	 Loss: 5.0143327713012695
    Step [150/192]	 Loss: 4.902953147888184
    Epoch [4/500]	 Loss: 4.947697018583615	 lr: 0.14998
    Step [0/192]	 Loss: 4.82228946685791
    Step [50/192]	 Loss: 4.764313220977783
    Step [100/192]	 Loss: 4.924652576446533
    Step [150/192]	 Loss: 4.818438529968262
    '
    '
    '
    Epoch [495/500]	 Loss: 3.9043560971816382	 lr: 4e-05
    Step [0/192]	 Loss: 3.893375873565674
    Step [50/192]	 Loss: 3.9129462242126465
    Step [100/192]	 Loss: 3.8850409984588623
    Step [150/192]	 Loss: 3.884953498840332
    Epoch [496/500]	 Loss: 3.903852423032125	 lr: 2e-05
    Step [0/192]	 Loss: 3.869290828704834
    Step [50/192]	 Loss: 3.9251136779785156
    Step [100/192]	 Loss: 3.8720123767852783
    Step [150/192]	 Loss: 3.9006035327911377
    Epoch [497/500]	 Loss: 3.9030883188048997	 lr: 1e-05
    Step [0/192]	 Loss: 3.949983835220337
    Step [50/192]	 Loss: 3.929286479949951
    Step [100/192]	 Loss: 3.900035858154297
    Step [150/192]	 Loss: 3.9218673706054688
    Epoch [498/500]	 Loss: 3.9071918639043965	 lr: 1e-05
    Step [0/192]	 Loss: 3.8706607818603516
    Step [50/192]	 Loss: 3.879091501235962
    Step [100/192]	 Loss: 3.887544870376587
    Step [150/192]	 Loss: 3.8865678310394287
    Epoch [499/500]	 Loss: 3.903093626101812	 lr: 0.0



```python

```
