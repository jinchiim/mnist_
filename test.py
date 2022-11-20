import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# 정확도를 높이기 위해 RandomForest 데려옴.


digits = load_digits() # sklearn에서 데려온 digits_data로 총 1797개, 64개의 픽셀, 클래스는 10개로 이루어짐  8 x 8형태

fig, axes = plt.subplots(ncols=4, nrows=4)
#fig = 그래프가 담기는 프레임이 되는 변수 (figure), axes(plot)= 데이터가 담기는 캔버스 ncols = 열의 수 , nrows = 행의 수
shuffle_list = list(range(1797))
# 0 ~ 1797번 shuffle_list 변수에 담음.
shuffle_list = random.choices(shuffle_list, k=4*4)
# shuffle_list에 있는 값중 가로4, 세로4를 랜덤으로 뽑아 shuffle_list에 넣어줌
for i, v in enumerate(shuffle_list):
    # enumerate: shuffle_list에 있는 값과 그 순서를 i, v에 넣어 전달 및 열거
    axes.flat[i].imshow(digits["images"][v], cmap="gray")
    # digits_data 안에 있는 images라는 키와 list값에 들어가 있는 v를 출력, cmap(colormap)은 "gray" 회색
    axes.flat[i].axis("off")
    # 캔버스 없앰
    axes.flat[i].set_title(f"Label {digits['target'][v]}")
    #  digits_data 안에있는 매치되는 숫자인 target를 데려와서 title에 부착

fig.tight_layout()
# subplot으로 fig 사이의 값 최소한의 여백이 생기게 함.
plt.show()
# 보여주는 함수

X = digits["data"]
# load_digits에 있는 "data" 이미지. 즉, 특성 행렬
y = digits["target"]
# load_digits에 있는 "target" 각 이미지에 대한 레이블 이라고 생각.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, shuffle=False
) # X, y로 나눈 데이터를 train_set와 test_set으로 분할, train_set 70프로, test_set = 30%, 
# random_state: 동일한 세트 생성하기 위해 주어짐, shuffle = False: 체계적 추출
X_train = np.array(X_train)
X_test = np.array(X_test)
# 모델 학습에 필요한 값은 numpy.array 형식 필수 list > numpy.array
X_train = X_train / 255
X_test = X_test / 255
# 모델은 정수 처리 보다는 실수 처리 할때의 성능이 더 좋으므로 0~255 의 픽셀값으로 이루어진 이미지를 0~1 픽셀값으로 바꿈

# 인공신경망 생성

random_forest = RandomForestClassifier(random_state=55)
random_forest.fit(X_train, y_train)

#예측해보기
y_pred = random_forest.predict(X_test)

# 분석결과 확인
random_report = classification_report(y_test, y_pred)
print(random_report)

# 오차행렬인 컴퓨전 메트릭스 확인
random_matrix = confusion_matrix(y_test, y_pred)
print(random_matrix)

#정확도 비교해보기
random_accuracy = accuracy_score(y_test, y_pred)
print('랜덤포레스트의 정확도 : ',random_accuracy)

pred_train_labels = random_forest.predict(X_train)
pred_test_labels = random_forest.predict(X_test)
# 예측치 
print("학습데이터셋", accuracy_score(y_train, pred_train_labels))
print("평가데이터셋", accuracy_score(y_test, pred_test_labels))


# 분류 정확도 

# TODO!: 손글씨 숫자 만들어서 분류기에 넣고 추론해보기
import cv2
px, py = -1, -1
def draw_mouse(event, x, y, flags, param):
    global px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        # 왼쪽 버튼이 눌릴때.
        px, py = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (px, py), (x, y), (255, 255, 255), 25, cv2.LINE_AA)
             # cv2. line = 255는 배경 색, 24는 선 굵기
            cv2.imshow("draw", img)
            px, py = x, y
            
# 마우스로 그린 것을 img로 저장

img = np.zeros((300, 300), dtype=np.uint8)
cv2.namedWindow("draw")
cv2.setMouseCallback("draw", draw_mouse, img)
cv2.imshow("draw", img)
cv2.waitKey()
cv2.destroyAllWindows()


img = cv2.resize(img, (8, 8))
# 8 * 8로 이미지 크기 변경 sklearn에서는 8 by 8의 이미지를 제공하기 때문.
img = img.flatten()
# 앞서 axes.flat 처럼 1차원 이미지로 변경
img = np.array(img)
# numpy.array 형식으로 변경
img = img / 255
# 정수가 아닌 실수로 변경
print(img)
pred_my_number = random_forest.predict([img])
# clf.predict에 img를 집어 넣어 학습된 X_train과 비교
print("예상한 값은 :%d" % pred_my_number)
####

cm = confusion_matrix(y_test, pred_test_labels)
sns.heatmap(cm, annot=True, fmt="g", vmin=0)
plt.ylabel("Target")
plt.xlabel("Prediction")
#plt.show()