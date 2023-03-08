---
layout: post
title:  SimCLR (Linear evaluate)
date:   2023-03-05 16:40:16
description: SimCLR 선형회귀평가 결과입니다.
tags: SimCLR
categories: experiment
---





앞서 보유하고 있는 데이터셋(5%, 25%, 50%)으로 Pretrained model을 만들었습니다.

훈련된 Pretrained model 뒤에 MLP레이어를 추가해 한번더 학습한 뒤 모델을 평가한 결과입니다.

[평가에 사용한 코드 링크]: https://yeongjin96.github.io/blog/2023/simCLR_50-_Eval/



평가에 사용한 파라미터는 다음과 같습니다.

model: ResNet50

optimizer : Adam

learning rate : 0.0003

loss function : CrossEntropy (Pytorch)

batch_size : 64

epochs : 100 (Test결과에는 epochs중 loss가 가장 낮은 가중치를 사용했습니다.)



class는 총 3개이며, train셋과 validation셋은 각 비율로 나눠진 데이터셋의 8:2 비율로 사용했습니다.



### 5%

##### SimCLR pretrained

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_5%.png" title="SimCLR_5%_Loss" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_5%_lr.png" title="SimCLR_5%_lr" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



##### Linear Evaluate

<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/5%_acc.png" title="5%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train(노랑) & Validation(보라)
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/5%_loss.png" title="5%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train(파랑) & Validation(분홍)
    </div>
</div>



###### Loss: 0.5545

###### Accuracy: 0.7559



### 25%

##### SimCLR pretrained

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_25%.png" title="SimCLR_25%_Loss" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_25%_lr.png" title="SimCLR_25%_lr" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



##### Linear Evaluate

<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/25%_acc.png" title="25%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train(노랑) & Validation(보라)
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/25%_loss.png" title="25%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train(파랑) & Validation(분홍)
    </div>
</div>



###### Loss: 0.3569

###### Accuracy: 0.8480



### 50%

##### SimCLR pretrained

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_50%.png" title="SimCLR_50%_Loss" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/pretrained/SimCLR_50%_lr.png" title="SimCLR_50%_lr" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



##### Linear Evaluate

<div class="col">
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/50%_acc.png" title="50%_acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    ACC: Train(노랑) & Validation(보라)
    </div>
    <div class="col-lg mt-2 mt-md-0">
        {% include figure.html path="assets/img/SimCLR/downstream/50%_loss.png" title="50%_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="caption">
    Loss: Train(파랑) & Validation(분홍)
    </div>
</div>



###### Loss: 0.3602

###### Accuracy: 0.8526
