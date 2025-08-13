# 퀵 정렬은 다음과 같은 과정으로 정렬을 진행한다. 정렬은 오름차순으로 진행한다고 가정하자
# 주어진 배열에서 하나의 요소를 선택하고 이를 pivot(피벗) 으로 삼는다.
# 배열 내부의 모든 값을 검사하면서 피벗 값보다 작은 값들은 왼쪽에, 큰 값들은 오른쪽에 배치한다.
# 이렇게 하면 배열이 두 부분으로 나뉜다. 나뉜 이 두 개의 배열에서 각각 새로운 피벗을 만들어서 두개의 배열로 다시 쪼개어 준다.
# 더 이상 배열을 쪼갤 수 없을 때까지 진행한다.
# 이 과정은 분할 정복의 원리를 이용한 것이다. 피벗을 중심으로 문제를 분할 하고, 
# 피벗을 기준으로 해서 작은 값과 큰 값을 나열하는 정복 과정을 거친 뒤, 모든 결과를 결합 해서 큰 전체 문제를 해결한다.

def quick_sort(a):
    n = len(a)

    if n <= 1:  # 종료조건
        return a
    
    # 기준값
    pivot = a[-1]   # 일반적으로 집합의 마지막 값을 기준
    g1 = []
    g2 = []
    for i in range(0, n-1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g2.append(a[i])
    return quick_sort(g1) + [pivot] + quick_sort(g2)

d = [6, 8, 3, 1, 2, 4, 7, 5]
print(quick_sort(d))



print("=================")

def quick_sort_sub(a, start, end):
    if end - start <= 0:
        return
    pivot = a[end]
    i = start
    for j in range(start, end):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]     # i 자리에 옮겨주고 i를 한 칸 뒤로
            i += 1
    a[i], a[end] = a[end], a[i]
    # 재귀 호출
    quick_sort_sub(a, start, i-1)   #  왼쪽 부분 정렬
    quick_sort_sub(a, i+1, end)

def quick_sort2(a):
    quick_sort_sub(a, 0, len(a)-1) 

d = [6, 8, 3, 1, 2, 4, 7, 5]
quick_sort2(d)
print(d)
