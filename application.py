from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from private_tool import *
import book_manager
app = Flask(__name__)

def imread(imgpath, reader):
    img_array = np.fromfile(imgpath, np.uint8)
    img = cv2.imdecode(img_array, reader)
    return img


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

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
    

def search_title(queue):
    titles = []
    data = loadJson(f"{path}/json/book_data.json")
    for key in data.keys():
        n = key.find(queue)
        if n+1:
            titles.append((key, n))
    titles = sorted(titles, key=lambda x:(x[1], x[0]))
    return titles
    
def search_book(uploaded: str, isbn :int or str):
    img = imread(path+f'/static/uploads/{uploaded}', cv2.IMREAD_COLOR)
    # template = cv2.imread('../img/taekwonv1.jpg')
    # template = imread(f'{path}/static/books/{isbn}/spine.jpg', cv2.IMREAD_COLOR)
    template = imread(f'D:/crawl_book_image/images/{isbn}/spine.jpg', cv2.IMREAD_COLOR)
    th, tw = template.shape[:2]
    # cv2.imshow('template', template)

    # 3가지 매칭 메서드 순회
    methods = ['cv2.TM_SQDIFF_NORMED'] #['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED',]
    img_draw = img.copy()
    for i, method_name in enumerate(methods):
        method = eval(method_name)
        # 템플릿 매칭   ---①
        try:
            res = cv2.matchTemplate(img, template, method)
        except:
            return "error.png"
        # 최솟값, 최댓값과 그 좌표 구하기 ---②
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(method_name, min_val, max_val, min_loc, max_loc)
        # TM_SQDIFF의 경우 최솟값이 좋은 매칭, 나머지는 그 반대 ---③
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            match_val = min_val
        else:
            top_left = max_loc
            match_val = max_val
        # 매칭 좌표 구해서 사각형 표시   ---④      
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),10)
        # 매칭 포인트 표시 ---⑤
        cv2.putText(img_draw, str(match_val), top_left, \
                    cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    imwrite(f"{path}/static/match/{uploaded[:-4]}_match.jpg", img_draw)
    return f"{uploaded[:-4]}_match.jpg"









#업로드 HTML 렌더링
@app.route('/', methods = ['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        print(request.form.keys())
        if "ISBN" in request.form.keys():
            add_result = book_manager.add_book_by_isbn(request.form["ISBN"])
            update_result = book_manager.update(request.form["ISBN"])
            return render_template('upload.html', add_book=add_result, update_data=update_result)
        if "queue" in request.form.keys(): #제목을 입력한 경우
            queue = request.form['queue']
            titles = search_title(queue)
            print(queue)
            return render_template('upload.html', search=True, titles=titles, queue=queue)
        else: #submit
            if not(request.form["title"]): #제목을 선택하지 않은 경우
                return render_template('upload.html', search=False, titles=[],title_fail=True)
            elif not(request.files["file"].filename): #파일을 업로드 하지 않은 경우
                title = request.form["title"]
                titles = search_title(title)
                return render_template('upload.html', search=True, queue=title, titles=titles, upload_fail=True)
            title = str(request.form["title"])
            titles = search_title(title)
            if "(" in title: #네이버 api 오류 방지용. 괄호 들어가는 제목은 인식이 잘 안되는 듯...
                title = title[:title.index("(")]
            isbn = search_book_isbn(title)[0]['isbn']
            f = request.files['file']
            uploaded = secure_filename(f.filename)
            f.save(path + '/static/uploads/' + uploaded)
            matched_img = search_book(uploaded, isbn)
            return render_template('upload.html', search=True, queue=title, titles=titles, img=matched_img)
    else: #시작 페이지
	    return render_template('upload.html')


    
#서버 실행
if __name__ == '__main__':  
    path = os.path.dirname(os.path.realpath(__file__))
    app.run(host='0.0.0.0', port=80, debug=True)



