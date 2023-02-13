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
####5%
Pretext-Loss

![5%_Pre_Loss](https://raw.githubusercontent.com/YeongJin96/YeongJin96.github.io/master/assets/img/5%_Pre_Loss.png)

Pretext-Accuracy

![5%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\5%_Pre_Acc.png)



DownStream-Loss

![5%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\5%_Down_Loss-1676248398766-11.png)

DownStream-Accuracy

![5%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\5%_Down_Acc.png)

####Test

Loss 0.7344 / Acc 64.11%



</details>

<details>
<summary>5%, lr=5e-6</summary>

####5%, lr=5e-6

Pretext-Loss

![Rot2_5%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_5%_Pre_Loss.png)

Pretext-Accuracy

![Rot2_5%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_5%_Pre_Acc.png)



DownStream-Loss

![Rot2_5%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_5%_Down_Loss.png)

DownStream-Accuracy

![Rot2_5%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_5%_Down_Acc.png)

####Test

Loss 0.5962 / Acc 83.88%



</details>

<details>
<summary>25%</summary>

##### 25%

Pretext-Loss

![25%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\25%_Pre_Loss.png)

Pretext-Accuracy

![25%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\25%_Pre_Acc.png)



DownStream-Loss

![25%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\25%_Down_Loss.png)

DownStream-Accuracy

![25%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\25%_Down_Acc.png)

####Test

Loss 0.3881 / Acc 84.51%



</details>

<details>
<summary>25%, lr=5e-6</summary>

####25%, lr=5e-6

Pretext-Loss

![Rot2_25%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_25%_Pre_Loss.png)

Pretext-Accuracy

![Rot2_25%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_25%_Pre_Acc.png)



DownStream-Loss

![Rot2_25%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_25%_Down_Loss.png)

DownStream-Accuracy

![Rot2_25%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_25%_Down_Acc.png)

####Test

Loss 0.3759 / Acc 90.91%



</details>

<details>
<summary>50%</summary>

##### 50%

Pretext-Loss

![50%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\50%_Pre_Loss.png)

Pretext-Acc

![50%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\50%_Pre_Acc.png)



DownStream-Loss

![50%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\50%_Down_Loss.png)

DownStream-Acc

![50%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\50%_Down_Acc.png)

####Test

Loss 0.3864 / Acc 85.76%



</details>



<details>
<summary>50%, lr=5e-6</summary>

##### 50%, lr=5e-6

Pretext-Loss

![Rot2_50%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_50%_Pre_Loss.png)

Pretext-Acc

![Rot2_50%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_50%_Pre_Acc.png)



DownStream-Loss

![Rot2_50%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_50%_Down_Loss.png)

DownStream-Acc

![Rot2_50%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot2_50%_Down_Acc.png)

####Test

Loss 0.3128 / Acc 87.78%



</details>



### RotNet+Jitter

<details>
<summary>5%</summary>

##### 5%

Pretext-Loss

![Rot+Jit_5%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_5%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit_5%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_5%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_5%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_5%_Down_Loss.png)

DownStream-Acc

![Rot+Jit_5%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_5%_Down_Acc.png)

####Test

Loss 0.7760 / Acc 79.32%



</details>



<details>
<summary>5%, lr=5e-6</summary>

##### 5%, lr=5e-6

Pretext-Loss

![Rot+Jit2_5%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_5%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_5%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_5%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_5%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_5%_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_5%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_5%_Down_Acc.png)

####Test

Loss 0.7794 / Acc 79.16%



</details>



<details>
<summary>25%</summary>

##### 25%

Pretext-Loss

![Rot+Jit_25%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_25%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit_25%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_25%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_25%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_25%_Down_Loss.png)

DownStream-Acc

![Rot+Jit_25%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_25%_Down_Acc.png)

####Test

Loss 0.3466 / Acc 86.64%



</details>



<details>
<summary>25%, lr=5e-6</summary>

##### 25%, lr=5e-6

Pretext-Loss

![Rot+Jit2_25%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_25%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_25%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_25%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_25%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_25%_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_25%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_25%_Down_Acc.png)

####Test

Loss 0.4978 / Acc 86.38%



</details>



<details>
<summary>50%</summary>

##### 50%

Pretext-Loss

![Rot+Jit_50%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_50%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit_50%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_50%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit_50%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_50%_Down_Loss.png)

DownStream-Acc

![Rot+Jit_50%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit_50%_Down_Acc.png)

####Test

Loss 0.2897 / Acc 89.40%



</details>



<details>
<summary>50%, lr=5e-6</summary>

##### 50%, lr=5e-6

Pretext-Loss

![Rot+Jit2_50%_Pre_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_50%_Pre_Loss.png)

Pretext-Acc

![Rot+Jit2_50%_Pre_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_50%_Pre_Acc.png)



DownStream-Loss

![Rot+Jit2_50%_Down_Loss](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_50%_Down_Loss.png)

DownStream-Acc

![Rot+Jit2_50%_Down_Acc](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Rot+Jit2_50%_Down_Acc.png)

####Test

Loss 0.3934 / Acc 90.80%



</details>



##### Test

| Loss/Acc | Rotation        | Rotation+Jitter | SimCLR |
| -------- | --------------- | --------------- | ------ |
| 5%       | 0.7344 / 64.11% | 0.7760 / 79.32% |        |
| 25%      | 0.3881 / 84.51% | 0.3466 / 86.64% |        |
| 50%      | 0.3864 / 85.76% | 0.2897 / 89.40% |        |

lr_schedular를 사용하지 않고 고정 lr = 0.000005로 학습시

| Loss/Acc | Rotation        | Rotation+Jitter | SimCLR |
| -------- | --------------- | --------------- | ------ |
| 5%       | 0.5962 / 83.88% | 0.7794 / 79.16% |        |
| 25%      | 0.3759 / 90.91% | 0.4978 / 86.38% |        |
| 50%      | 0.3128 / 87.78% | 0.3934 / 90.80% |        |



#Supervised 학습시 

100%데이터를 사용했을때, loss: 0.3376 / Acc: 88.96%의 결과가 나왔습니다.

(데이터 5%만 사용해 학습시 loss: 2.824 / Acc: 53.43%)