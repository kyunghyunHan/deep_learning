import torch
import torch.nn as nn# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))

# 합성곱층
conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)


pool = nn.MaxPool2d(2)
print(pool)


out = conv1(inputs)
print(out.shape)


out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)
