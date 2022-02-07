 # korean_food_classifier


![image](https://user-images.githubusercontent.com/33340741/152799667-1145bd0c-23b9-4461-9248-379079d8f119.png)

[한국음식이미지 데이터](https://aihub.or.kr/aidata/13594)를 이용한 이미지 분류기 모델


## Dataset

tf.data.Dataset 을 이용한 데이터 파이프라인 구축 : [kfood_dataset.py](https://github.com/kimhwijin/korean_food_classifier/blob/master/kfood_dataset.py)

## Model

|모델 이름|구조|파라미터|훈련 에폭|정확도|
|---|---|---|---|---|
|KerasInceptionResNetV2|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/KerasInceptionResNetV2)|54,567k|20|20%|
|CustomInceptionResNetV2|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/inception_resnet_v2.py)|54,567k|40|93%|
|CustonInceptionResNetV2 + SEBlock|[Structure](https://github.com/kimhwijin/korean_food_classifier/blob/master/application/inception_resnet_v2_se.py)|62,923k|40|67%|


