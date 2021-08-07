# R-CNN, SPPNet

Created At: 2021년 7월 23일 오전 10:05
Created By: 조승제
Topics: Deep Learning, Object Detection
Type: 📒 Lesson
발표일: 2021년 7월 23일
발표자: 조승제
참석자: 유정민, 엄현식, 조건우

# Background

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%201.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%201.png)

- IoU (Intersection Over Union)
    - Bounding Box를 얼마나 잘 예측하였는지를 판단하는 지표
    - 교집합 / 합집합

        ![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.28.39.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.28.39.png)

- mAP

    : AP 값의 평균

    - AP

        ![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.37.47.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.37.47.png)

        - precision = TP / (TP + FP)
        - recall = TP / (TP + FN)
        - precision, recall 값은 confidence 값의 threshold 값에 따라 달라짐

        - threshold 값을 0 ~ 1.0까지 0.1 단위로 증가시키면서 precision, recall 값을 계산

            ![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.41.27.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.41.27.png)

        - AP = 그래프 선의 아래쪽 면적으로 계산

- Bounding Box

    ![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-19__3.59.31.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-19__3.59.31.png)

- NMS

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.48.03.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.48.03.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.45.37.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/_2021-07-20__6.45.37.png)

# R-CNN (2014)

## 1. Introduction

- The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT and HOG
- It is generally acknowledged that progress has been slow during 2010-2012
- 기존 방법들은 비효율적 → CNN 도입 → 기존 연구들보다 월등한 성능 개선

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%202.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%202.png)

# 2. Object detection with R-CNN

## 1) Generates category-independent region proposals.

## 2) A large convolutional neural network that extracts a fixed-length feature vector from each region.

## 3) Classification을 위한 linear SVMs

### 2.1) Region Proposals.

- 기존에 category-independent region proposals 연구가 활발했음
    - ex) objectness, selective search, category-independent object proposals, CPMC 등등
- 본 논문에선 Selective Search 기법을 사용
- 2000개의 독립적인 region proposal 생성

Selective Search

: Segmentation 분야에 많이 쓰이는 알고리즘이며, 객체와 주변 간의 색감, 질감 차이 등을 파악해서 물체 위치를 파악

1. 이미지의 초기 segment를 지정하고, region 영역 생성
2. greedy 알고리즘을 이용해서 각 region을 기준으로 주변의 유사한 영역을 결합
3. 결합된 region을 최종 region proposal로 제안

### 2.2) Feature Extraction

- Pre-trained CNN Model (AlexNet) 사용
- 서로 다른 크기를 가진 region proposal들을 warp → (227, 227) RGB pixel

warp
(x, y) 좌표의 픽셀을 (x', y') 좌표로 대응시키는 작업

[참고](https://nostudy.tistory.com/27)

- 2000개의 region proposal들을 CNN을 통과시켜 4096차원의 feature vector를 추출
- CNN → 5개의 conv layer → 2개의 fully connected layer → feature vector 추출

### 2.3) Test-time detection

- Fully connected layer를 통과한 feature들은 SVM을 통해 각 class로 분류됨
- SVM을 통과한 region proposal은 NMS를 적용하여 하나의 bounding box만 남김
- NMS를 적용하여 IoU가 가장 높은 bounding box를 선택
- 이후 bounding box regression 적용 → ground-truth box와 비슷하게 조정

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%203.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%203.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%204.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%204.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%205.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%205.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%206.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%206.png)

## 3. Training

### Supervised pre-training

- CNN 모델은 ILSVRC2012 classification 데이터셋으로 훈련

### Domain-specific fine-tuning

- CNN을 detection task와 warped proposal windows에 적응시키기 위해, warped region proposals을 사용하여 SGD로 50000번 훈련시킴

### Object category classifiers

- Ground truth box → positive sample, IoU 값이 0.3 보다 작은 것은 negative sample로 지정 (IoU 값이 0.3보다 큰 경우 무시)
- positive sample 32개, negative sample 96개, 총 128개의 mini-batch를 구성
- mini-batch → CNN → 4096 차원 feature vector 추출
- 추출된 벡터로 linear SVMs training (hard negative mining 기법 적용)

Hard negative mining

사람을 탐지하면 positive sample, 배경을 탐지하면 negative sample

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%207.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%207.png)

## 4. Experiments

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%208.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%208.png)

---

# SPPNet (2015)

## 1. Introduction

- R-CNN의 문제점
    - 1개의 이미지에 대해 2000번의 CNN을 수행 → 시간적 비용 손해
    - Selective Search 이후 wrap 과정에서 이미지 왜곡 발생 → 성능 저하 가능성
    - CNN에 227x227의 고정된 input이 필요한 이유에 대한 의문
        - In fact, convolutional layers do not require a fixed image size and can generate feature maps of any sizes.
        - On the other hand, the fully-connected layers need to have fixedsize/length input by their definition.

- spatial pyramid pooling (SPP) layer to remove the fixed-size constraint of the network
- Specifically, we add an SPP layer on top of the last convolutional layer.

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%209.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%209.png)

- The SPP layer pools the features and generates fixedlength outputs, which are then fed into the fullyconnected layers (or other classifiers). → 처음부터 이미지를 crop, warp 하지 않아도 되게 됨

## 2. The Spatial Pyramid Pooling Layer

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2010.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2010.png)

[코드 참고](https://github.com/gunooknam/SPPNet/blob/master/sppnet.py)

- Bag-of-Words(BoW)에서 파생

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2011.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2011.png)

- (?, ?, 256)의 feature map이 들어오면 filter size, stride 값을 조절해서 Max Pooling
- 위 사진에서는 {4x4, 2x2, 1x1}(totally 21 bins) 적용
- SPP layer 덕분에 input image의 크기에 제한 없이 특징을 잘 반영할 수 있게 되었음
- 또한 pooling은 resolution을 감소시키는데, 여러 pooling을 수행하면서 다양한 resolution을 가지게 됨

## 3. Experiments

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2012.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2012.png)

## 4. Conclusion

- SPP를 통해 RCNN에서의 warping 작업을 없애서 이미지 왜곡을 없앰
- RCNN은 CNN 연산을 2000번 했지만, SPP에서는 1번만 하면서 train, test 시간 단축

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2013.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2013.png)

![R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2014.png](R-CNN,%20SPPNet%20e0c39fa81470462bb56dcf257adffdb0/Untitled%2014.png)

---

## 의문

- [x]  CNN fine tuning 과정?
- For VOC, N = 20 and for ILSVRC2013, N = 200. We treat all region proposals with ≥ 0.5 IoU overlap with a ground-truth box as positives for that box’s class and the rest as negatives.
- SVM과 비슷하게 진행됨, IoU값이 0.5 이상이면 Positive sample, 이하이면 negative sample

- [x]  SVM이 왜 좋은 성능을 내는가?
- Network의 Overfitting을 피하기 위해 Positive 데이터가 많아야 하는데, 시가상으로 데이터가 많지 않아 softmax classifier을 적용했을 때 성능이 좋지 않았음

- [x]  BoW랑 무슨 관계?
- BoW → Bag of Visual Words → Spatial Pyramid Matching