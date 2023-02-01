---
layout: post
title: (SSL)1.Data_split
date: 2023-01-31 19:09:00
description: 실험에 사용할 data를 나눈 코드입니다.
tags: data_preprocess SSL
categories: code
---

1. 원본데이터중 x40(magnification) 이미지만 사용
2. train_test ratio를 7:3으로 함
3. dataset의 imbalance를 막기 위해, dataset의 크기가 가장 작은 클래스를 중점으로 class당 dataset크기를 재조정
4. train 데이터를 100%, 70%, 50%, 30%만 사용하도록 분할


```python
import os
from glob import glob
import shutil
import numpy as np
import random
random.seed(42)
```


```python
ori_path = "/home/yj/workspace/SimCLR/data/PNU/"
```


```python
for (root, dirs, files) in os.walk(ori_path):
    print("# root : " + root)
    if len(dirs) > 0:
        for dir_name in dirs:
            print("dir: " + dir_name)

    if len(files) > 0:
        for i in range(3):
            print("file: " + files[i])
```

    # root : /home/yj/workspace/SimCLR/data/PNU/
    dir: Atypical Ductal Hyperplasia
    dir: Ductal Carcinoma In-Situ
    dir: Normal
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia
    dir: S15-3188_B
    dir: S14-7332_B
    dir: S11-3325_B
    dir: S11-8600_B
    dir: S15-9038_B
    dir: S11-4889_B
    dir: S12-8459_B
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia/S15-3188_B
    dir: x5
    dir: x40
    dir: x10
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia/S15-3188_B/x5
    file: S15-3188_B x5 (116).png
    file: S15-3188_B x5 (138).png
    file: S15-3188_B x5 (109).png
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia/S15-3188_B/x40
    file: S15-3188_B x40(31).png
    file: S15-3188_B x40(1534).png
    file: S15-3188_B x40(720).png
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia/S15-3188_B/x10
    file: S15-3188_B x10 (42).png
    file: S15-3188_B x10 (53).png
    file: S15-3188_B x10 (287).png
    # root : /home/yj/workspace/SimCLR/data/PNU/Atypical Ductal Hyperplasia/S14-7332_B
    dir: x5
    dir: x40
    dir: x10
    '
    '
    '
    # root : /home/yj/workspace/SimCLR/data/PNU/Normal/S14-8521_A/x10
    file: S14-8521_A x10 (13).png
    file: S14-8521_A x10 (32).png
    file: S14-8521_A x10 (10).png



```python
class_list = os.listdir(ori_path)
print(class_list)

class_path = []
for i in range(len(class_list)):
    class_path.append(ori_path+"/"+class_list[i])
    print("path:",class_path[-1])
```

    ['Atypical Ductal Hyperplasia', 'Ductal Carcinoma In-Situ', 'Normal']
    path: /home/yj/workspace/SimCLR/data/PNU//Atypical Ductal Hyperplasia
    path: /home/yj/workspace/SimCLR/data/PNU//Ductal Carcinoma In-Situ
    path: /home/yj/workspace/SimCLR/data/PNU//Normal



```python
def Extract_name(path):
    tmp = path.split("/")
    
    return tmp[-1]
```

dataset의 train_test 비율을 셔플 후, 8:2로 나눔


```python
def Train_test_split(ori_path, class_path, save_path, ratio=0.8):
    train_path = save_path+"train"
    test_path = save_path+"test"
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    print("save_path: ", save_path)
    
    class_ = []
    for class_ in class_path:
        class_name = Extract_name(class_)
        os.makedirs(train_path+"/"+class_name, exist_ok=True)
        os.makedirs(test_path+"/"+class_name, exist_ok=True)
        glob_file = glob(ori_path+class_name+"/**/x40/**.png", recursive=True)
        random.shuffle(glob_file)
        
        total_size = len(glob_file)
        train_size = round(len(glob_file) * ratio)
        test_size = total_size - train_size
        
        for i in range(len(glob_file)):
            img_name = Extract_name(glob_file[i])
            
            if i < train_size:
                shutil.copy2(glob_file[i], train_path+"/"+class_name+"/"+img_name)
            else:
                shutil.copy2(glob_file[i], test_path+"/"+class_name+"/"+img_name)
                
        print("{}'s dataset_size\n train: {}\n test: {}\n total: {}".format(class_name, train_size, test_size, total_size))
        
```


```python
# def Shutil_copy(new_path, min_size, save_path):
#     compare_list = []
#     for class_ in new_path:
#         class_name = Extract_name(class_)
#         os.makedirs(save_path+class_name, exist_ok=True)
#         glob_file = glob(new_path+class_name+"/**/**.png", recursive=True)
#         random.shuffle(glob_file)
        
#         for i in range(min_size):
#             img_name = Extract_name(glob_file[i])
#             shutil.copy2(glob_file[i], save_path+class_name+"/"+img_name)
```

dataset에서 test_dataset을 분리후, 나머지 train_dataset중 가장 적은 클래스 data_size와 다른 클래스들의 data_size를 동일하게 함


```python
def Compare_size(train_path):
    class_list = os.listdir(train_path)
    
    compare_list = []
    for class_ in class_list:
        compare_list.append(len(glob(train_path+class_+"/*.png")))
        
    return np.min(compare_list)
```


```python
save_path = "../data/PNU/"
Train_test_split(ori_path, class_path, save_path)
```

    save_path:  ../data/PNU/
    Atypical Ductal Hyperplasia's dataset_size
     train: 7505
     test: 1876
     total: 9381
    Ductal Carcinoma In-Situ's dataset_size
     train: 3079
     test: 770
     total: 3849
    Normal's dataset_size
     train: 4542
     test: 1136
     total: 5678



```python
train_path = save_path+"train/"
readjust_train_path = "../data/train_%100"
min_size = Compare_size(train_path)

for class_ in class_list:
    img_path_list = glob(train_path+class_+"/*.png")
    os.makedirs(readjust_train_path+"/"+class_, exist_ok=True)
    for i in range(min_size):
        img_name = Extract_name(img_path_list[i])
        shutil.copy2(img_path_list[i], readjust_train_path+"/"+class_+"/"+img_name)
```

재조정된 train_data중 30%, 50%, 70%만 사용하기 위해, 각각의 데이터셋을 새로 만듦


```python
def Ratio_split(train_path, ratio):
    new_path = "../data/train_%" + str("%g" %(ratio*100))
    os.makedirs(new_path, exist_ok=True)
    
    class_list = os.listdir(train_path)
    for class_ in class_list:
        file_list = glob(train_path+"/"+class_+"/*.png")
        random.shuffle(file_list)
        num_data = round(len(file_list) * ratio)
        
        os.makedirs(new_path+"/"+class_, exist_ok=True)
        for i in range(num_data):
            img_name = Extract_name(file_list[i])
            shutil.copy2(file_list[i], new_path+"/"+class_+"/"+img_name)
```


```python
Ratio_split(readjust_train_path, 0.3)
Ratio_split(readjust_train_path, 0.5)
Ratio_split(readjust_train_path, 0.7)
```


```python
print("data_30% size:",len(glob("../data/train_%30/**/**.png", recursive=True)))
print("data_50% size:",len(glob("../data/train_%50/**/**.png", recursive=True)))
print("data_70% size:",len(glob("../data/train_%70/**/**.png", recursive=True)))
print("data_100% size:",len(glob("../data/train_%100/**/**.png", recursive=True)))
```

    data_30% size: 2772
    data_50% size: 4620
    data_70% size: 6465
    data_100% size: 9237



```python

```
