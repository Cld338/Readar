import requests
from bs4 import BeautifulSoup
import os
import json
import urllib
import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from private_tool import *

def getCover(soup):
    cover_url = str(soup.select_one("#CoverMainImage"))
    try:
        cover_url = cover_url[cover_url.index("src=")+5:].replace("\"/>","")
        return cover_url
    except:
        return None

def getSpine(soup, coverUrl):
    try:
        # print(soup.select("head > meta > meta"))
        spine = str(soup.select("head > meta > meta")[7]).split()[-2][:-1]
    except:
        spine = coverUrl.replace("cover500", "spineflip")[:-6]+"_d.jpg"
        return spine
    spine = coverUrl[:coverUrl.index("cover500")]+"spineflip/"+spine+"_d.jpg"
    return spine

def getISBN(soup):
    ISBN = str(soup.select_one("head > meta > link > link"))
    ISBN = ISBN.replace("<link href=\"http://aladin.kr/p/", "").replace("\" rel=\"shortlink\"/>", "")
    if len(ISBN)!=10 and len(ISBN)!=13:
        return None
    return ISBN    

def search_title(queue):
    titles = []
    data = loadJson(f"{currDir}/json/book_data.json")
    for key in data.keys():
        n = key.find(queue)
        if n+1:
            titles.append((key, n))
    titles = sorted(titles, key=lambda x:(x[1], x[0]))
    return titles


def add_book_by_isbn(isbn):
    book_ls = filesInFolder(f"{currDir}/static/books/images")
    if str(isbn) in book_ls:
        return "이미 등록된 책 입니다."
    session = requests.session()
    retry = Retry(connect=1, backoff_factor=0)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        res = session.get(f"http://aladin.kr/p/{isbn}")
    except:
        return "오류가 발생했습니다."
    soup = BeautifulSoup(res.text, 'html.parser')
    cover_url = getCover(soup)
    if cover_url == None or "img_no" in cover_url:
        return
    spine_url = getSpine(soup, cover_url)
    if spine_url==None or isbn==None:
        return "아직 이미지가 등록되지 않은 책입니다."
    if requests.get(spine_url).status_code != 200:
        return "서버 오류"
    createDirectory(f"{currDir}/static/books/images/{isbn}")
    urllib.request.urlretrieve(cover_url, f"{currDir}/static/books/images/{isbn}/cover.jpg")
    urllib.request.urlretrieve(spine_url, f"{currDir}/static/books/images/{isbn}/spine.jpg")

















def search_book_isbn(query):
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode, quote
    import json

    CLIENT_ID = "8ZuLkjROEOhJwLyGdrzo"
    CLIENT_SECRET = "P1WYLIJpBa"

    request = Request('https://openapi.naver.com/v1/search/book?query='+quote(query))
    request.add_header('X-Naver-Client-Id', CLIENT_ID)
    request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)
    response = urlopen(request).read().decode('utf-8')
    search_result = json.loads(response)
    return search_result['items']
    
def update(ISBN):
    original_data = loadJson(f"{currDir}/json/book_data.json")
    original_isbn = [i["ISBN"] for i in original_data.values()]
    if ISBN in original_isbn:
        return ""
    try:
        data = search_book_isbn(ISBN)[0]
    except:
        return "네이버 api에 등록되지 않은 책입니다."
    title = data["title"]
    publisher = data["publisher"]
    pubyear = data["pubdate"][:4]
    original_data[title] = {
        "puplisher":publisher,
        "pubyear":pubyear,
        "ISBN":ISBN
    }
    saveJson(f"{currDir}/json/book_data.json", original_data)