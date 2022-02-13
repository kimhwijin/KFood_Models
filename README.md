 # korean_food_classifier


![image](https://user-images.githubusercontent.com/33340741/152799667-1145bd0c-23b9-4461-9248-379079d8f119.png)
![image](https://user-images.githubusercontent.com/33340741/152803530-02949121-2f13-4ecc-9d0a-7565a6e975e1.png)

[한국음식이미지 데이터](https://aihub.or.kr/aidata/13594)를 이용한 이미지 분류기 모델

## 데이터셋 정보
범주 : 총 150 개의 음식종류 [Label, Class](https://github.com/kimhwijin/korean_food_classifier/blob/master/class_to_label.txt) 

이미지 : png, jpg, jpeg, gif, bmp 형식의 다양한 이미지
- 각 범주당 1000개의 이미지, 총 15만개의 이미지가 포함된다.

구조 :

```
압축 해제 전

-kfood.zip
--구이.zip
--...

후

-kfood
--구이
---갈비구이
----crop_area.properties
----Img_000_0000.jpg
----...
---...
--...

```
##### 훈련 세트 : 70%, 테스트 세트 : 20%, 검증 세트 : 10%

## Dataset

tf.data.Dataset 을 이용한 데이터 파이프라인 구축 : [kfood_dataset.py](https://github.com/kimhwijin/korean_food_classifier/blob/master/kfood_dataset.py)

### 대용량 데이터를 훈련하기 위한 데이터 전처리 방식
1. 모든 데이터의 경로들을 모아 데이터셋으로 생성한다.
2. 데이터경로를 통해 이미지를 로드한다.
3. 이미지 전처리를 수행하고 레이블을 지정한다.
4. shuffle, batch, repeat, prefetch 을 지정한다.

### 이미지 전처리 구성
1. 한국음식 데이터에 포함되어있는 crop_area.properties 의 크롭 정보를 통해 이미지를 자른다.
2. Random Crop : 이미지를 랜덤하게 90% 축소시키면서 자른다.
3. Central Crop : 이미지의 너비와 높이가 동일하도록 중앙을 기준으로 자른다.
4. 위의 Crop 중 한가지를 수행후, 이미지 사이즈 299x299 가 되도록 Resize 한다.
5. 이미지 픽셀 값이 0 ~ 1 이 되도록 float32 로 변환후 정규화 한다.
6. 레이블은 One-Hot Encoding 을 수행한다.


## Model

|모델 이름|구조|파라미터|훈련 에폭|정확도|
|---|---|---|---|---|
|KerasInceptionResNetV2|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/keras_inception_resnet_v2.py)|54,567k|-|-|
|InceptionResNetV2|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/inception_resnet_v2.py)|30,627k|-|-|
|KerasInceptionResNetV2SEBlock|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/keras_inception_resnet_v2_se.py)|60,696k|-|-|


## Optimizer

- SGD


|모델|정확도(테스트 세트)|Learning Rate|Momentum|Nesterov|Learning Rate Decay|
|---|---|---|---|---|---|
|KerasInceptionResNetV2|-|0.01|0.9|True|0.001(linear)|
|InceptionResNetV2|-|0.01|0.9|True|0.001(linear)|
|KerasInceptionResNetV2SEBlock|-|0.01|0.9|True|0.001(linear)|

- RMSprop


|모델|테스트세트 정확도(훈련 세트)|Epochs|Learning Rate|Decay(rho)|Momentum|Epsilon|Learning Rate Decay|
|---|---|---|---|---|---|---|---|
|KerasInceptionResNetV2|-|-|0.045|0.9|0.0|1.0|0.94(exp, per 2 epochs)|
|KerasInceptionResNetV2|73%(99.67%)|50|0.001|0.9|0.9|1.0|-|
|InceptionResNetV2|89.7%(96%)|70|0.045|0.9|0.0|1.0|0.94(exp, per 2 epochs)|
|KerasInceptionResNetV2SEBlock|-|-|0.045|0.9|0.0|1.0|0.94(exp, per 2 epochs)|

- Decay
  - initial learning rate : 0.045, exp decay per 2 epochs : 0.9

![output](https://user-images.githubusercontent.com/33340741/153397885-8706b1a8-6bc7-4dc3-9b90-f351e70f77d0.png)
