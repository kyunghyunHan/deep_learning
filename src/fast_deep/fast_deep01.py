from itertools  import permutations

arr = ['A','B','C']
#원소중에서 2개를 뽑는 모든 순열 계산
result =list(permutations(arr,2))
print(result)