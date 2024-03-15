import torch
import torch.nn.functional as F

torch.manual_seed(1)

z= torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z,dim =0)

print(hypothesis)
print(hypothesis.sum())

z= torch.rand(3,5,requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)


y = torch.randint(5, (3,)).long()
print(y)


# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y.unsqueeze(1))
print(y_one_hot)


torch.log(F.softmax(z,dim=1))
print(torch.log(F.softmax(z,dim=1))
)
print(F.log_softmax(z, dim=1)
)

#4개가 같음
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()# 위와 같음
F.nll_loss(F.log_softmax(z, dim=1), y) # 위와 같음 log_softmax를 수행한다음 나음 수식들 수행
F.cross_entropy(z, y)
