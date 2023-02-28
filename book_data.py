import json
import requests
import time
from private_tool import *


def search_book_isbn(query):
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode, quote
    import json

    CLIENT_ID = "ID"
    CLIENT_SECRET = "KEY"

    request = Request('https://openapi.naver.com/v1/search/book?query='+quote(query))
    request.add_header('X-Naver-Client-Id', CLIENT_ID)
    request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)
    response = urlopen(request).read().decode('utf-8')
    search_result = json.loads(response)
    return search_result['items']
    

ISBNls = filesInFolder(f"{currDir}/static/books/images")
original_data = loadJson(f"{currDir}/json/book_data.json")
original_isbn = [i["ISBN"] for i in original_data.values()]
for ISBN in ISBNls:
    if ISBN in original_isbn:
        continue    
    print(ISBN)
    data = search_book_isbn(ISBN)[0]
    title = data["title"]
    publisher = data["publisher"]
    pubyear = data["pubdate"][:4]
    original_data[title] = {
        "puplisher":publisher,
        "pubyear":pubyear,
        "ISBN":ISBN
    }
    saveJson(f"{currDir}/json/book_data.json", original_data)