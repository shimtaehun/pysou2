# 선택 정렬(selection sort) 은 주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로
# 알고리즘 과정:
# 최소값 찾기: 정렬되지 않은 부분에서 가장 작은 값을 찾습니다.
# 교환: 찾은 최소값을 정렬되지 않은 부분의 맨 앞으로 이동시킵니다.
# 반복: 정렬되지 않은 부분의 크기가 1이 될 때까지 위 과정을 반복합니다.

# 방법 1) 원리 이해를 우선
def find_minFunc(a):
    n = len(a)
    min_idx = 0
    for i in range(1, n):
        if a[i] < a[min_idx]:
            min_idx = i
    return min_idx

def sel_sort(a):
    result = []
    while a:
        min_idx = find_minFunc(a)
        value = a.pop(min_idx)
        result.append(value)
    return result

d = [2, 4, 5, 1, 3]
# print(find_minFunc(d))
print(sel_sort(d))



# 방법 2) 일반적 정렬 알고리즘을 구사 : result X
# 각 반복마다 가장 작은 값을 해당 집합내의 맨 앞자리와 값을 바꿈

def sel_sort2(a):
    n = len(a)
    for i in range(0, n - 1):   # 0부터  n-2까지 반복
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]

d = [2, 4, 5, 1, 3]
print(sel_sort2(d))