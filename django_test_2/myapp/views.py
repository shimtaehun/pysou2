from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd

# Create your views here.
def indexFunc(request):
    return render(request, 'index.html')

def dbshowFunc(request):
    sql = """
        select j.jikwonno as 사번, j.jikwonname as 직원명,
        b.busername as 부서명, j.jikwonjik as 직급,
        j.jikwonpay as 연봉, j.jikwonibsail as 근무년수,
        j.jikwongen as 성별
        from jikwon j inner join buser b
        on j.busernum = b.buserno
    """
    params = []
    if dept:
        sql += " where b.busername like %s" # + dept를 쓰면 해킹에 취약해서 따음표안에 %s를 넣는다.
        params.append(f"%{dept}%")      #   SQL 해킹방식 (시큐어 코딩)
    sql += " order by j.jikwonno, j.jikwonname"

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        print('cur.description: ', cur.description)
        cols = [c[0] for c in cur.description]
        # print('cols: ', cols)

    df = pd.DataFrame(rows, columns=cols)
    # print(df.head(3))

    # join 결과로 html 생성
    if not df.empty:
        join_html = df[['사번', '직원명', '부서명', '직급','연봉', '근무년수','성별']].to_html(index=False)
    else:
        join_html = '조회된 자료가 없어요'
    
    # 직급별 연봉 통계표 (NaN -> 0 처리)
    if not df.empty:
        stats_df = (
            df.groupby("직급")["연봉"]
                .agg(평균="mean", 표준편차=lambda x:x.std(ddof=0), 인원수="count")
                .round(2)
                .reset_index()
                .sort_values(by="평균", ascending=False)
        )
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        stats_html = stats_df.to_html(index = False)
    else:
        stats_html = "통계 대상 자료가 없어요"

    ctx_dict = {
        'dept':escape(dept), # 문자열에 특수문자가 있는 경우 HTML엔티티로 치환하는(단순문자취급) 열할을 함, 해킹을  방지하기 위해서 사용
                            # 예) escape('<script>alert(1)</script> -> '&lt;script&gt;....)
        'join_html': join_html,
        'stats_html': stats_html,
    }
    return render(request, 'dbshow.html', ctx_dict)