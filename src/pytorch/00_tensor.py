import numpy as np
import torch

# pytorch
## 1차원
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
##  shape나 size()를 사용하면 크기를 확인
print(t)
print(t.dim())  # rank. 즉, 차원
print(t.shape)  # shape
print(t.size()) # shape

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱

# 2차원
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
print(t.dim())  # rank. 즉, 차원
print(t.size()) # shape
print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
print(t[:, 1].size()) # ↑ 위의 경우의 크기
print(t[:, :-1])


## broadcasting
## 행렬 A,B가 있다고 가정 - 덧셈과 뺼셈을할떄는 크기가 같아야함
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)
## 브로드캐스팅은 편리하지만  크기를 잘보고 사용해야함

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)




## 곱셈
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

## element_wise곱셈
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))


## 평균
t=  torch.FloatTensor([1,2])
print(t.mean())

## 2차원
t= torch.FloatTensor([[1,2],[3,4]])
print(t)
print(t.mean())
print(t.mean(dim=0))

## dim=0 => 1번째 차원 
# 실제 연산 과정
##t.mean(dim=0)은 입력에서 첫번째 차원을 제거한다.

## [[1., 2.],
 ##[3., 4.]]

## 1과 3의 평균을 구하고, 2와 4의 평균을 구한다.
## 결과 ==> [2., 3.]


print(t.mean(dim=1))


# 실제 연산 결과는 (2 × 1)
##[1. 5]
##[3. 5]

## 덧셈
t= torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거


## 최대 Max와 아그맥스
t= torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.max())
print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])