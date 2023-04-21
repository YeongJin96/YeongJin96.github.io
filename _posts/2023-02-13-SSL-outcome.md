---
layout: post
title: SSL 실험결과(임시)
date: 2023-02-12 15:09:00
description: Self Supervisied Learn모델 실험 결과입니다.
tags: SSL
categories: Experimentd
---



### Unsupervised Representation Learning by Predicting Image Rotations

<details>
<summary>5%</summary>

##### 5%

Pretext-Loss

![5%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/5%25_Pre_Loss-1676252081192-2.png)

Pretext-Accuracy

![5%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/5%25_Pre_Acc.png)



DownStream-Loss

![5%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/5%25_Down_Loss-1676248398766-11.png)

DownStream-Accuracy

![5%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/5%25_Down_Acc.png)

##### Test

Loss 0.7344 / Acc 64.11%



</details>

<details>
<summary>5%, lr=5e-6</summary>

##### 5%, lr=5e-6

Pretext-Loss

![Rot2_5%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_5%25_Pre_Loss.png)

Pretext-Accuracy

![Rot2_5%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_5%25_Pre_Acc.png)



DownStream-Loss

![Rot2_5%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_5%25_Down_Loss.png)

DownStream-Accuracy

![Rot2_5%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_5%25_Down_Acc.png)

##### Test

Loss 0.5962 / Acc 83.88%



</details>

<details>
<summary>25%</summary>

##### 25%

Pretext-Loss

![25%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/25%25_Pre_Loss.png)

Pretext-Accuracy

![25%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/25%25_Pre_Acc.png)



DownStream-Loss

![25%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/25%25_Down_Loss.png)

DownStream-Accuracy

![25%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/25%25_Down_Acc.png)

##### Test

Loss 0.3881 / Acc 84.51%



</details>

<details>
<summary>25%, lr=5e-6</summary>

##### 25%, lr=5e-6

Pretext-Loss

![Rot2_25%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_25%25_Pre_Loss.png)

Pretext-Accuracy

![Rot2_25%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_25%25_Pre_Acc.png)



DownStream-Loss

![Rot2_25%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_25%25_Down_Loss.png)

DownStream-Accuracy

![Rot2_25%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_25%25_Down_Acc.png)

##### Test

Loss 0.3759 / Acc 90.91%



</details>

<details>
<summary>50%</summary>

##### 50%

Pretext-Loss

![50%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/50%25_Pre_Loss.png)

Pretext-Acc

![50%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/50%25_Pre_Acc.png)



DownStream-Loss

![50%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/50%25_Down_Loss.png)

DownStream-Acc

![50%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/50%25_Down_Acc.png)

##### Test

Loss 0.3864 / Acc 85.76%



</details>



<details>
<summary>50%, lr=5e-6</summary>

##### 50%, lr=5e-6

Pretext-Loss

![Rot2_50%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_50%25_Pre_Loss.png)

Pretext-Acc

![Rot2_50%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_50%25_Pre_Acc.png)



DownStream-Loss

![Rot2_50%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_50%25_Down_Loss.png)

DownStream-Acc

![Rot2_50%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot2_50%25_Down_Acc.png)

##### Test

Loss 0.3128 / Acc 87.78%



</details>



### RotNet+Jitter

<details>
<summary>5%</summary>

##### 5%

Pretext-Loss

![Rot+Jit_5%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_5%25_Pre_Loss-1676253045630-5.png)

Pretext-Acc

![Rot+Jit_5%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_5%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_5%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_5%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit_5%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_5%25_Down_Acc.png)

##### Test

Loss 0.7760 / Acc 79.32%



</details>



<details>
<summary>5%, lr=5e-6</summary>

##### 5%, lr=5e-6

Pretext-Loss

![Rot+Jit2_5%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_5%25_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_5%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_5%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_5%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_5%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_5%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_5%25_Down_Acc.png)

##### Test

Loss 0.7794 / Acc 79.16%



</details>



<details>
<summary>25%</summary>

##### 25%

Pretext-Loss

![Rot+Jit_25%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_25%25_Pre_Loss.png)

Pretext-Acc

![Rot+Jit_25%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_25%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_25%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_25%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit_25%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_25%25_Down_Acc.png)

##### Test

Loss 0.3466 / Acc 86.64%



</details>



<details>
<summary>25%, lr=5e-6</summary>

##### 25%, lr=5e-6

Pretext-Loss

![Rot+Jit2_25%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_25%25_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_25%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_25%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_25%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_25%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_25%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_25%25_Down_Acc.png)

##### Test

Loss 0.4978 / Acc 86.38%



</details>



<details>
<summary>50%</summary>

##### 50%

Pretext-Loss

![Rot+Jit_50%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_50%25_Pre_Loss.png)

Pretext-Acc

![Rot+Jit_50%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_50%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_50%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_50%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit_50%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit_50%25_Down_Acc.png)

##### Test

Loss 0.2897 / Acc 89.40%



</details>



<details>
<summary>50%, lr=5e-6</summary>

##### 50%, lr=5e-6

Pretext-Loss

![Rot+Jit2_50%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_50%25_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_50%_Pre_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_50%25_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_50%_Down_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_50%25_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_50%_Down_Acc](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/Rot_outcome/Rot%2BJit2_50%25_Down_Acc.png)

##### Test

Loss 0.3934 / Acc 90.80%



</details>

<br>

##### Test (Rotation & Rotation + Jitter)

Encoder인 ResNet50의 학습가중치를 고정하지 않고(unfreeze) MLP와 함께 학습해 얻은 Rotation과 Rotation+Jitter의 결과입니다.

Rotation의 경우 원본이미지를 0°, 90°, 180°, 270° 회전시킨 뒤, Pretext task에서 예측하도록 학습한 뒤, Downstream에서 label을 주고 학습하였습니다.

Rotation+Jitter의 경우 Rotation Pretext task에 각 회전마다 다른 Jitter augmentation을 적용시켜 회전시킨 이미지를 예측하도록 학습한 뒤, Downstream에서 label을 주고 학습하였습니다.

| Loss/Acc | Rotation        | Rotation+Jitter |
| -------- | --------------- | --------------- |
| 5%       | 0.7344 / 64.11% | 0.7760 / 79.32% |
| 25%      | 0.3881 / 84.51% | 0.3466 / 86.64% |
| 50%      | 0.3864 / 85.76% | 0.2897 / 89.40% |

lr_schedular를 사용하지 않고 고정 lr = 0.000005로 학습시

| Loss/Acc | Rotation        | Rotation+Jitter |
| -------- | --------------- | --------------- |
| 5%       | 0.5962 / 83.88% | 0.7794 / 79.16% |
| 25%      | 0.3759 / 90.91% | 0.4978 / 86.38% |
| 50%      | 0.3128 / 87.78% | 0.3934 / 90.80% |

<br>

<br>

Encoder인 ResNet50의 학습가중치를 고정하고(freeze) Downstream에서 MLP만 학습해 얻은 Rotation과 Rotation+Jitter의 결과입니다.

##### Test (Rotation & Rotation + Jitter)

| Loss/Acc | Rotation (Encoder freeze) | Rotation+Jitter (Encoder freeze) |
| -------- | ------------------------- | -------------------------------- |
| 5%       | 0.9604 / 50.28%           | 0.7794 / 79.16%                  |
| 25%      | 0.7235 / 67.99%           | 0.7713 / 64.60%                  |
| 50%      | 1.0940 / 38.56%           | 0.7783 / 62.57%                  |

<br>

<br>

SimCLR으로 학습한 결과입니다.

1. ImageNet의 가중치를 가져와 보유하고있는 의료 데이터셋(유방암)으로 Pretext task를 진행하였습니다.<br>
2. ImageNet의 가중치 없이 유방암 패치가 아닌 신장 패치데이터(약 15만장)로 Pretext task후, 보유하고 있는 데이터로 mlp를 학습하였습니다.<br>
3. ImageNet의 가중치를 가져와 신장 패치데이터를 Pretext task후, 보유하고 있는 데이터(유방암)로 mlp를 학습하였습니다.<br>
4. 2와 같이 가중치 없이 신장 데이터로 Pretext task후, 학습된 가중치에 보유하고 있는 데이터셋(유방암)으로 한번 더 Pretext task를 진행한 뒤, 보유하고 있는 데이터로 mlp를 학습하였습니다.<br>
5. 3과 같이 ImageNet가중치를 가져와 신장데이터로 Ptrext task후, 보유하고 있는 데이터셋(유방암)으로 한번 더 Pretext task를 진행한 뒤, 보유하고 있는 데이터로 mlp를 학습하였습니다.<br>

##### Test (SimCLR)

| Loss/Acc | 1. SimCLR (dataset + ImageNet) | 2. SimCLR (kidneys dataset) | 3. SimCLR (kidneys dataset + ImageNet) | 4. (kidneys+dataset) | 5. (kidneys+ImageNet+dataset) |
| -------- | ------------------------------ | --------------------------- | -------------------------------------- | -------------------- | ----------------------------- |
| 5%       | 0.5545 / 75.59%                | 0.6162 / 73.62%             | 0.7177 / 69.64%                        | 0.5309 / 76.97%      | 0.4881 / 80.65%               |
| 25%      | 0.3569 / 84.80%                | 0.5261 / 77.87%             | 0.5176 / 79.77%                        | 0.3193 / 87.50%      | 0.2785 / 88.96%               |
| 50%      | 0.2767 / 88.79%                | 0.4094 / 83.55%             | 0.4538 / 82.79%                        | 0.2686 / 89.46%      | 0.2229 / 91.50%               |


#Supervised 학습시 

| Data size | Loss   | Accuracy |
| --------- | ------ | -------- |
| 5%        | 1.595  | 67.10%   |
| 25%       | 0.5947 | 83.15%   |
| 50%       | 0.5538 | 88.70%   |
