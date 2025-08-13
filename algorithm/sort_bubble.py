# 버블 정렬은 인접한 두 개의 원소를 비교하여 자리를 교환하는 방식이다.
# [2, 4, 5, 1, 3]
#  2와 4의 값을 비교 4와 5의 값을 비교 5와 1을 비교해서 자리 변경
# 2 4 1 5 3
# 2 1 4 5 3
# 1 2 4 5 3
# 1 2 4 3 5
# 1 2 3 4 5
# 방식으로 

def bubble_sort(a):
    n = len(a)
    while True:
        changed = False     # 자료를 바꾸었는지 여부
        for i in range(0 , n - 1):
            if a[i] > a[i + 1]:     #  앞이 뒤보다 크면
                print(a)
                a[i], a[i+1] = a[i + 1], a[i]
                changed = True
        if changed == False:
            return


d = [2, 4, 5, 1, 3]
bubble_sort(d)
print(d)