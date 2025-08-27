from django.shortcuts import render
from django.db import connection

def dictfetchall(cursor):
    """
    커서(cursor)로부터 실행 결과를 가져와 딕셔너리 리스트로 반환합니다.
    템플릿에서 사용하기 편리한 형태입니다.
    """
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def jikwon_list_view(request):
    sql = """
        select j.jikwonno as "직원번호", j.jikwonname as "직원명",
        b.busername as "부서명", b.busertel as "부서전화",
        j.jikwonpay as "연봉", j.jikwonjik as "직급",
        j.jikwonibsail as "입사일"
        from jikwon j inner join buser b
        on j.busernum = b.buserno
    """
    
    with connection.cursor() as cursor:
        cursor.execute(sql)
        jikwon_list = dictfetchall(cursor) # 결과를 딕셔너리 리스트로 변환

    return render(request, 'jikwon_list.html', {'jikwon_list': jikwon_list})