---
layout: post
title: SimCLR 학습결과
date: 2023-01-18 19:09:00
description: SimCLR 모델을 Epochs와 Batch 그리고 Optimizer에 차이를 두고 학습한 결과입니다.
tags: SimCLR
categories: Experiment
---



SimCLR 모델을 Epochs와 Batch 그리고 Optimizer에 차이를 두고 학습한 결과입니다.

학습에 사용한 Parameters는 다음과 같습니다.

<br/>

<br/>

<br/>


### Augmentation
torchvision.transforms 라이브러리를 사용해 Augmentation 했고, 코드는 다음과 같습니다.
<br/>
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

<br/>


Color_Jitter([

**brightness**, **contrast**, **saturation**, **hue** = 0.8

]) #해당 값들을 (1-0.8) ~ (1+0.8)의 범위를 적용확률 0.8로 augmentation

<br/>


**Resized&Crop** = 0.08 ~ 1.0의 비율중 랜덤하게 Crop한 후 0.75 ~ 1.3333의 비율로 너비와 높이를 줄이거나 늘린후 원본 이미지 size로 만듬

<br/>


**HorizontalFlip** = Random(0.5)으로 이미지를 Flip

<br/>


**Grayscale** = Random(0.2)으로 이미지를 Grayscale 

<br/>


해당 방법으로 데이터셋 이미지 1개를 각각 다른이미지로 Augmentation 후 학습.
<br/>
<br/>
<br/>

### Model & Batch, Optimizer
<br/>
SimCLR 학습에 사용한 encoder모델은 **ResNet18**과 **ResNet50**을 사용했습니다.

Batch **64, 128, 256**에 나누어 진행했고, **LARS**와 **Adam** Optimizer를 각각 적용했습니다.

<br/>

<br/>

<br/>



### Outcome

<br/>


SimCLR_train code: [Train코드 링크](https://yeongjin96.github.io/blog/2023/SimCLR_train/)

SimCLR_evaluate code: [Evaluate코드 링크](https://yeongjin96.github.io/blog/2023/simCLR_evaluate/)

<br/>

<br/>

#### ResNet18 (LARS)

<br/>

| 데이터 갯수 | Batch_size |  Loss  | Accuracy |
| :---------: | :--------: | :----: | :------: |
|     300     |     64     | 0.629  |  0.7103  |
|     300     |    128     | 0.6260 |  0.7209  |
|     300     |    256     | 0.6140 |  0.7251  |
|     500     |     64     | 0.6206 |  0.7310  |
|     500     |    128     | 0.5709 |  0.7502  |
|     500     |    256     | 0.5472 |  0.7619  |
|    1000     |     64     | 0.5495 |  0.7622  |
|    1000     |    128     | 0.5280 |  0.7678  |
|    1000     |    256     | 0.4706 |  0.8032  |
|    1500     |     64     | 0.4944 |  0.7943  |
|    1500     |    128     | 0.4276 |  0.8152  |
|    1500     |    256     | 0.3936 |  0.8362  |

<br/>

<br/>

<br/>

<br/>

<br/>



#### Reference

[https://github.com/Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR)
