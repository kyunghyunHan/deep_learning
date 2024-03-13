import numpy as np
import torch
"""
- 1차원 벡터
- 2차원 행렬
- 3차원이상->텐서
"""

# 1차원
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
#  shape나 size()를 사용하면 크기를 확인
print(t)
print(t.dim())  # rank. 즉, 차원
print(t.shape)  # shape
print(t.size()) # shape
#슬라이싱
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



"""
broadcasting
- 덧셈과 뺼셈을 할떄는 두 행렬 A,B가 같아야함
- 자동으로 크기를 맞춰서 연산을 수행하게 만드는 브로드캐스팅
"""
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



"""
mul과  matmul의 차이
- matmul = 행렬곱셈
"""
## 곱셈
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

## element_wise곱셈
## broad casting후 계산
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
"""
[1]
[2]
==> [[1, 1],
     [2, 2]]
->broad casting
"""
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
"""
[[1., 2.],
 [3., 4.]]
"""
## 1과 3의 평균을 구하고, 2와 4의 평균을 구한다.
## 결과 ==> [2., 3.]


print(t.mean(dim=1))


# 실제 연산 결과는 (2 × 1)
#[1. 5]
#[3. 5]

# 덧셈
t= torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거


# 최대 Max와 아그맥스
# 최대(Max)는 원소의 최대값을 리턴하고, 아그맥스(ArgMax)는 최대값을 가진 인덱스를 리턴
t= torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.max())
print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
"""
Max:  tensor([3., 4.])
Argmax:  tensor([1, 1])
"""


"""
뷰
- 원소의 수를 유지하면서 텐서의 크기변경
"""
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
## torch.Size([2, 2, 3])

print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
print(ft.view([-1, 3]).shape)## -1은 사용자가 잘모르겟으니 파이토치가 알아서 3은 두번쨰 길이의 차원의길이는 3
## view는 기본적으로 변경전과 변경후의 텐서안의  원소의 개수가 유지되어야함


## 3차원 텐서의 크기변경
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
## Squeeze - 1인 차원을 제거
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())## 차원축소
print(ft.squeeze().shape)## 

## Unsqueeze - 특정 위치에 1인 차원을 추가
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.view(1, -1))
print(ft.view(1, -1).shape)

##  타입캐스팅
lt= torch.LongTensor([1,2,3,4])
print(lt)

print(lt.float())

bt= torch.ByteTensor([True,False,False,True])
print(bt)
print(bt.long())

## concat

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))
"""
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])

        tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
"""

## stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))


## zeros_like
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기

print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
"""
tensor([[2., 4.],
        [6., 8.]])
tensor([[1., 2.],
        [3., 4.]])

"""
## 덮어쓰기 연산
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # 기존의 값 출력
"""
tensor([[2., 4.],
        [6., 8.]])
tensor([[2., 4.],
        [6., 8.]])
"""