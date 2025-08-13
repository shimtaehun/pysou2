# JSON : XML에 비해 가벼우며, 배열에 대한 지식만 있으면 처리 가능

import json

dict = {'name':'tom','age':33, 'score':['90', '80', '100']}
print("dict:%s"%dict)
print(type(dict))

print('json encoding(dict를 Json 모양의 문자열로 변경하는 것---')
str_val = json.dumps(dict)
print("str_val:%s"%str_val)
print(type(str_val))

print('json decoding(str을 dict로 변경하는것)---')
json_val = json.loads(str_val)
print("json_val:%s"%json_val)
print(type(json_val))
print(json_val['name'])

for k in json_val.keys():
    print(k)

print("웹에서 json 문서 읽기----------")

import urllib.request as req
url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"
plainText = req.urlopen(url).read().decode()
print(plainText)
print(type(plainText))
jsonData = json.loads(plainText)
print(type(jsonData))
print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME'])     #LH강남3단지작은도서관

# dict의 자료를 읽어 도서관명, 전화, 주소
libData = jsonData.get('SeoulLibraryTime').get('row')
# print(libData)
print(libData[0].get('LBRRY_NAME'))     # LH강남3단지작은도서관

datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADDRS')
    # print(name, tel, addr)
    datas.append([name, tel, addr])

import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명', '전화번호', '주소'])
print(df)