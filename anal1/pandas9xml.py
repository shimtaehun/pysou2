# xml 문서 처리
from bs4 import BeautifulSoup

with open('anal1/my.xml', mode = 'r', encoding='utf-8') as f:
    xmlfile = f.read()
    # print(xmlfile)

soup = BeautifulSoup(xmlfile, 'lxml')
# print(soup.prettify())
itemTag = soup.findAll('item')
# print(itemTag)
print(itemTag[0])
print()
nameTag = soup.find_all('name')
print(nameTag[0]['id'])

print('---------------')
for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        print('id:' + j['id'] + ', name:' + j.string)
        tel = i.find('tel')
        print('tel:' + tel.string)
    for j in i.findAll('exam'):
        print('kor:' + j['kor'] + ', eng:' + j['eng'])
    print()
    
