---
layout: post
title:  SimCLR (Linear evaluate)
date:   2023-03-05 16:40:16
description: SimCLR 선형회귀평가 결과입니다.
tags: SimCLR
categories: experiment
---


<br>
<br>

앞서 보유하고 있는 데이터셋(5%, 25%, 50%)으로 Pretrained model을 만들었습니다.

훈련된 Pretrained model 뒤에 MLP레이어를 추가해 한번더 학습한 뒤 모델을 평가한 결과입니다.

<a href="https://yeongjin96.github.io/blog/2023/simCLR_50-_Eval">평가에 사용한 코드 링크</a>
<br>
<br>

### Dataset

---

데이터셋은 배율(Magnification) **x5**, **x10**, **x40**으로 나누어져 있고, 각 배율마다 **Benign** / **Atypical Ductal Hyperplasia (ADH)** / **Ductal Carcinoma In-Situ (DCIS)** 3개의 Class로 분리되어 있습니다. 

이중 제가 사용한 데이터의 배율은 x40입니다.  
각 클래스별 데이터의 수는 다음과 같습니다.

{% include figure.html path="assets/img/SimCLR/pretrained/Dataset_Table.jpg" title="Dataset_Table" class="img-fluid rounded z-depth-1" zoomable=true %}



이후 Train dataset & Test dataset을 8:2 비율로 Split했습니다.

| Original Dataset            | Train | Test |
| --------------------------- | ----- | ---- |
| Benign                      | 4542  | 1136 |
| Atypical Ductal Hyperplasia | 7505  | 1876 |
| Ductal Carcinoma In-Situ    | 3079  | 770  |
| Total                       | 15126 | 3782 |


각 클래스별로 데이터 사이즈의 편차가 커서, 가장 작은 데이터사이즈의 클래스와 나머지 클래스들의 데이터 사이즈를 동일하게 사용하였습니다.

| Resized Dataset             | Dataset | Test |
| --------------------------- | ------- | ---- |
| Benign                      | 3079    | 1136 |
| Atypical Ductal Hyperplasia | 3079    | 1876 |
| Ductal Carcinoma In-Situ    | 3079    | 770  |
| Total                       | 9237    | 3782 |

---


평가에 사용한 파라미터는 다음과 같습니다.
<br>
model: ResNet50

optimizer : Adam

learning rate : 0.0003

loss function : CrossEntropy (Pytorch)

batch_size : 64

epochs : 100 (Test결과에는 epochs중 loss가 가장 낮은 가중치를 사용했습니다.)

<br>
<br>

class는 총 3개이며, train셋과 validation셋은 각 비율로 나눠진 Train데이터셋의 8:2 비율로 사용했습니다.

<br>
<br>
<br>

### SimCLR pretrained

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/loss.png" title="Pretext_task_Loss" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_5%_lr.png" title="Pretext_task_lr" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<br>
<br>

---

<br>
### 5%
<br>

##### Dataset Table

| Dataset for Pretext Task    | Train |
| --------------------------- | ----- |
| Benign                      | 154   |
| Atypical Ductal Hyperplasia | 154   |
| Ductal Carcinoma In-Situ    | 154   |
| Total                       | 462   |

<br>

| Dataset for Evaluate        | Train | Validation | Test |
| --------------------------- | ----- | ---------- | ---- |
| Benign                      | 126   | 28         | 1136 |
| Atypical Ductal Hyperplasia | 122   | 32         | 1876 |
| Ductal Carcinoma In-Situ    | 122   | 32         | 770  |
| Total                       | 370   | 92         | 3782 |

<br>

##### Linear Evaluate

<br>

<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_5%_acc.png" title="5%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train & Validation
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_5%_loss.png" title="5%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train & Validation
    </div>
</div>
<br>

###### Loss: 0.5545
###### Accuracy: 0.7559

---

<br>
### 25%
<br>

##### Dataset Table

| Dataset for Pretext Task    | Train |
| --------------------------- | ----- |
| Benign                      | 770   |
| Atypical Ductal Hyperplasia | 770   |
| Ductal Carcinoma In-Situ    | 770   |
| Total                       | 2310  |

<br>

| Dataset for Evaluate        | Train | Validation | Test |
| --------------------------- | ----- | ---------- | ---- |
| Benign                      | 613   | 157        | 1136 |
| Atypical Ductal Hyperplasia | 621   | 149        | 1876 |
| Ductal Carcinoma In-Situ    | 614   | 156        | 770  |
| Total                       | 1848  | 462        | 3782 |

<br>

##### Linear Evaluate

<br>

<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_25%_acc.png" title="25%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train & Validation
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_25%_loss.png" title="25%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train & Validation
    </div>
</div>
<br>
###### Loss: 0.3569
###### Accuracy: 0.8480

---

<br>
<br>
<br>
### 50%

<br>

##### Dataset Table

| Dataset for Pretext Task    | Train |
| --------------------------- | ----- |
| Benign                      | 1540  |
| Atypical Ductal Hyperplasia | 1540  |
| Ductal Carcinoma In-Situ    | 1540  |
| Total                       | 4620  |

<br>

| Dataset for Evaluate        | Train | Validation | Test |
| --------------------------- | ----- | ---------- | ---- |
| Benign                      | 1247  | 293        | 1136 |
| Atypical Ductal Hyperplasia | 1243  | 297        | 1876 |
| Ductal Carcinoma In-Situ    | 1206  | 334        | 770  |
| Total                       | 3696  | 924        | 3782 |

<br>

##### Linear Evaluate

<br>
<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_50%_acc.png" title="50%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train & Validation
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/mat_50%_loss.png" title="50%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train & Validation
    </div>
</div>
<br>
###### Loss: 0.2767
###### Accuracy: 0.8879
