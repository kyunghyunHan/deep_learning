#순열
from itertools  import permutations

arr = ['A','B','C']
#원소중에서 2개를 뽑는 모든 순열 계산
result =list(permutations(arr,2))
print(result)

#조합
from itertools import combinations

arr = ['A','B','C']
#원소증에서 2개를 뽑는 모든 조합 계산
result = list(combinations(arr,2))
print(result)

#중복순열
from itertools import product

arr = ['A','B','C']
result= list(product(arr,repeat=2))
print(result)

#중복조합
from itertools import combinations_with_replacement

arr = ['A','B','C']

#원소중에서 2개를 뽑는  모든 중복 조합 계산
result= list(combinations_with_replacement(arr,2))
print(result)