---
layout: post
title: SimCLR params별 학습결과(수정중)
date: 2023-01-18 19:00:00
description: SimCLR모델의 parameter별 학습결과입니다.
tags: formatting code
categories: sample-posts
---

<h1>SimCLR의 파라미터별 학습결과</h1>



SimCLR 모델을 Epochs와 Batch 그리고 Optimizer에 차이를 두고 학습한 결과입니다.



학습에 사용한 Parameters는 다음과 같습니다.



#### Augmentation

torchvision.transforms 라이브러리를 사용해 Augmentation을 했고, 코드는 다음과 같습니다.

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
                torchvision.transforms.RandomHorizontalFlip(),
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

Color_Jitter([

**brightness, contrast, saturation, hue** = 0.8

]) #해당 값들을 (1-0.8) ~ (1+0.8)의 범위를 적용확률 0.8로 augmentation

**Resized&Crop** = 0.08 ~ 1.0의 비율중 랜덤하게 Crop한 후 0.75 ~ 1.3333의 비율로 너비와 높이를 줄이거나 늘린후 원본 이미지 size로 만듬

**HorizontalFlip** = Random(0.5)으로 이미지를 Flip

**Grayscale** = Random(0.2)으로 이미지를 Grayscale

해당 방법으로 데이터셋 이미지 1개를 각각 다른이미지로 Augmentation 후 학습.



#### Model & Batch, Optimizer

SimCLR 학습에 사용한 encoder(based_CNN)모델은 **ResNet18**과 **ResNet50**을 사용했습니다.

Batch **64, 128, 256**에 나누어 진행했고, **LARS와 Adam** Optimizer를 각각 적용했습니다.



#### Outcome

| Epochs | Batch_size | Optimizer | loss | accuracy |
| :----: | :--------: | :-------: | :--: | :------: |
|        |            |           |      |          |
|        |            |           |      |          |
|        |            |           |      |          |
|        |            |           |      |          |
|        |            |           |      |          |
|        |            |           |      |          |

