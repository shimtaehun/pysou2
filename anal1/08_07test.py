import requests
from bs4 import BeautifulSoup
import pandas as pd

# 스크래핑할 굽네치킨 메뉴 목록 URL
url = 'https://www.goobne.co.kr/menu/menu_list_p'

print(f"'{url}' 에서 메뉴를 가져오는 중입니다...")


response = requests.get(url)
response.raise_for_status() 
soup = BeautifulSoup(response.text, 'html.parser')
names = [tag.text.strip() for tag in soup.select('div.textbox > h4')]
print(names)
prices = [tag.text.replace('해당 가격은 권장 소비자가 입니다.', '').replace('\n', '').strip() 
for tag in soup.select('div.textbox > p')]
print(prices)
df = pd.DataFrame({'상품명':names, '가격':prices})
print(df.head(3))