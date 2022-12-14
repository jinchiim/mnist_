# mnist_
mnist를 이용한손글씨 분류기를 만들어 숫자 예측해보기.

**사용 프로그램**
* * *
python　　　　3.9   
matplotlib　　　　3.6.0   
numpy　　　　1.22.4   
scikit-learn　　　　1.1.3   
seaborn　　　　0.12.0   
* * * 

* 자세한 설명은 코드 밑 주석 참고 
* 정확도를 높이기 위해 random_forest 사용으로 0.93까지 도달    
* 모델은 정수 처리 보다는 실수 처리 할때의 성능이 더 좋으므로 0 ~ 255 의 픽셀값으로 이루어진 이미지를 0~1 픽셀값으로 바꿈
* img >  numpy.array 형식으로 변경
* 사진 크기 변경 하려 했으나, sklearn이 8 * 8 이미지 모델 제공 확인

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
더 많은 데이터로 더 많은 반복 학습 필요, 특정 패턴을 확인하는 CNN을 활용하면 정확도가 더 높다는 정보를 확인하였으나, 기존 손글씨로 비교하기 때문에 조금 더 일치한다 판단하는 숫자로 오류 생성되는 것이 아닌가 라는 판단을 내렸습니다.    
기존 띄워진 손글씨 숫자 이미지와 비슷하게 작성하면, 맞는 답을 도출하는 결과 확인할 수 있었습니다.

### 가설 도움 이미지 ###
***
**보여준 랜덤 손글씨 이미지**

<img src="./test1.png" width="70%" height="80%" title="test" alt="test"></img>


**결과**    
<img src="./result1.png" width="70%" height="80%" title="test" alt="result"></img>


