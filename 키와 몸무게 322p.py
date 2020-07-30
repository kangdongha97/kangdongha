


# 학습 데이터셋
X = [[170,80],[175,76],[180,70],[160,55],[163,43],[165,48]]
d = [1, 1, 1, 0, 0, 0]
# 가중치와 바이어스 초기값
w = [0.0, 0.0]
b = 0.0
l_rate = 0.1  # 학습률 
n_epoch = 50  # 에포크 횟수
for epoch in range(n_epoch): # 에포크 반복
    sum_error = 0.0
    for XX, dd in zip(X, d): # 데이터셋을 4번반복
        f = b   # 바이어스 기존값
        for i in range(2):  # 입력신호 총합 계산
            f += w[i] * XX[i]  # f(sum(wx)+b)
        if f >= 0.0:  # 스텝 활성화 함수
            y = 1.0
        else:
            y = 0.0
        error = dd - y  # 실제 출력 계산
        b = b + l_rate * error 
        sum_error += error**2 # 오류의 제곱 계산
        for i in range(2): # 가중치 변경
            w[i] = w[i] + l_rate * error * XX[i]
        print(w, b)
    print('에포크 =%d, 오류=%.3f' % (epoch, sum_error))
print(w, b)

tX = [168, 55]  # 테스트 데이터
f=0
for i in range(2): 
   f += w[i] * tX[i]
f+=b
print("f=",f)
if f >= 0.0:  
    y = 1
else:
    y = 0
print(y)    # 테스트 결과