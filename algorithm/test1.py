# 알고리즘은 문제를 해결하기 위한 일련의 단계적 절차 또는 방법을 의미한다.
# 즉, 어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어들을 순서대로 나열한 것
# 컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 합니다.
# 문제 -> 입력 -> 알고리즘으로 처리 -> 출력

# 문) 1 ~ 10(n) 까지의 정수의 합 구하기'
# 방법1 : O(n)
def totFunc(n):
    tot = 0
    for i in range(1, n + 1):
        tot = tot + i           # 더하기 n번
    return tot
print(totFunc(10000000000000000000000))

# 방법2: O(1)
def totFunc2(n):
    return n * (n+1) // 2       # 덧셈 후 곱셈 후 나눗셈
print(totFunc2(100))

# 주어진 문제를 푸는 방법은 다양하다. 어떤 방법이 더 효과적ㅇ니지 알아내는 것이 '알고리즘 분석'
# '알고리즘 분석' 평가 방법으로 계산 복잡도 표현 방식이 있다.
# 1) 공간 복잡도: 메모리 사용량 분석
# 2) 시간 복잡도: 처리 시간을 분석
# O(빅오) 표기법: 빅오 표기법은 알고리즘의 효율성을 표기해주는 표기법

#문2) 임의의 정수들 중 최대값 찾기
# 입력: 숫자 n개를 가진 list
# 최대값 찾기
# 출력: 숫자 n개 중 최대값
def findMaxFunc(a):
    n = len(a)
    max_v = a[0]
    for i in range(1, n):
        if a[i] > max_v:
            max_v = a[i]    # 최대값 변경
    return max_v


d = [17, 92, 18, 33, 58, 7, 33, 42] 
print(findMaxFunc(d))

# 최댓값 위치 반환
def findMaxFunc2(a):
    n = len(a)
    max_v = 0
    for i in range(1, n):
        if a[i] > a[max_v]:
            max_v = i    # 최대값 변경
    return max_v

d = [17, 92, 18, 33, 58, 7, 33, 42] 
print(findMaxFunc(d))

# 문3) 동명이인 찾기 : n명의 사람 이름 중 동일한 이름을 찾아 결과 출력
imsi = ['길동','순신','순신','길동']
imsi2 = set(imsi)
imsi = list(imsi2)
print(imsi)

def findsameFunc(a):
    n = len(a)
    result = set()
    for i in range(0, n-1): # 0부터 n-2까지 반복
        for j in range(i+1, n): # i+1부터 n-1까지 반복
            if a[i] == a[j]:    # 이름이 같으면
                result.add(a[i])
    return result
names = ['tom', 'jerry', 'mike', 'tom']
print(findsameFunc(names))

# 문4) 팩토리얼
# 방법1) for
def factFunc(n):
    imsi = 1
    for i in range(1, n+1):
        imsi = imsi * i
    return imsi

print(factFunc(5))
# 방법2) 재귀 호출
def factFunc2(n):
    if n <= 1:      # 종료 조건 필수
        return 1
    return n * factFunc2(n-1)   # 재귀 호출

print(factFunc2(5))

# 재귀 연습1) 1부터 n까지의 합 구하기: 재귀 사용
def sumFunc(n):
    if n <= 1:
        return 1
    return n + sumFunc(n-1)
print(sumFunc(10))

# 재귀 연습2) 숫자 n개 중 최대값 구하기: 재귀 사용
def maxFunc(a, n):
    if n == 1:
        return a[0]
    max_of_rest = maxFunc(a, n - 1)
    if a[n - 1] > max_of_rest:
        return a[n - 1]
    else:
        return max_of_rest
values = [7, 9, 15, 42, 33, 22]
print(maxFunc(values, len(values)))
