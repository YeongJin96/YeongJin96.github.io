---
layout: post
title: SSL 실험결과(임시)
date: 2023-02-07 15:09:00
description: Self Supervisied Learn모델 실험 결과입니다.
tags: SSL
categories: Experimentd
---













### Unsupervised Representation Learning by Predicting Image Rotations



##### 5%

Pretext-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/5%25_Pre_Loss.png)

Pretext-Accuracy



![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/5%25_Pre_Acc.png)

DownStream-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/5%25_Down_Loss.png)

DownStream-Accuracy

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/5%25_Down_Acc.png)

25%

Pretext-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/25%25_Pre_Loss.png)

Pretext-Acc

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/25%25_Pre_Acc.png)

DownStream-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/25%25_Down_Loss.png)

DownStream-Acc

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/25%25_Down_Acc.png)

50%

Pretext-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/50%25_Pre_Loss.png)

Pretext-Acc

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/50%25_Pre_Acc.png)

DownStream-Loss

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/50%25_Down_Loss.png)

DownStream-Acc

![](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/50%25_Down_Acc.png)

Test

| loss/Acc | Rotation        | Rotation+Jitter | SimCLR |
| -------- | --------------- | --------------- | ------ |
| 5%       | 0.7344 / 64.11% | 0.7760 / 79.32% |        |
| 25%      | 0.3881 / 84.51% | 0.3466 / 86.64% |        |
| 50%      | 0.3864 / 85.76% |                 |        |

#Supervised 학습시 

100%데이터를 사용했을때, loss: 0.3376 / Acc: 88.96%의 결과가 나왔습니다.

(데이터 5%만 사용해 학습시 loss: 2.824 / Acc: 53.43%)