# mnist_
mnist를 이용한손글씨 분류기를 만들어 숫자 예측해보기.

**사용 프로그램**
* * *
python                  3.9
matplotlib              3.6.0
numpy                   1.22.4
scikit-learn            1.1.3
seaborn                 0.12.0
* * * 

* 자세한 설명은 주석 참고 
* 정확도를 높이기 위해 random_forest 사용
* 모델은 정수 처리 보다는 실수 처리 할때의 성능이 더 좋으므로 0 ~ 255 의 픽셀값으로 이루어진 이미지를 0~1 픽셀값으로 바꿈
* img >  numpy.array 형식으로 변경

```python

random_forest = RandomForestClassifier(random_state=55)
random_forest.fit(X_train, y_train)
.
.
img = np.array(img)
.
.
img = img / 255
.
```


**왜 정확도가 높지 않은가?**
더 많은 데이터로 더 많은 반복 학습 필요, 기존 손글씨로 비교하기 때문에 조금 더 일치한다 판단하는 숫자로 오류 생성 가능
비슷하게 작성하면, 맞는 답을 도출하는 결과 확인.


